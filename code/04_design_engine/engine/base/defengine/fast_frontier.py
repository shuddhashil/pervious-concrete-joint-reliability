"""Fast frontier search (decision E-07). Run with the GPU python, ONE process:
  python -m defengine.fast_frontier --out runs/ff1 [--prior runs/gpu4 runs/explore4 ...] [--lo 200 --hi 500 --step 25] [--no-verify]
Order of work (all search stages are GPU; acceptance is the frozen CPU estimator, run in a separate process while the GPU keeps searching):
  0 archive    every design any earlier run ever found is re-scored at every cement level on ONE fresh bank -> an instant first frontier
  1 walk       deep solve at the lowest level that already looks good, then walk DOWN level by level, each level seeded with the best known designs;
               stop after two levels in a row that miss the target
  2 refine     5 kg then 1 kg levels between the last miss and the first hit; independent second search at the lowest hits
  3 verify     frozen estimator, fresh seeds, 2 replicates, top-2 designs per level, lowest levels first (wave 1 overlaps stage 2)
Search estimates never accept anything: a level counts only if a design passes BOTH verification replicates."""
import os, sys, json, glob, time, argparse, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENGINE_PY = ROOT.parent / "A3_engine_20261001" / "runtime" / "python.exe"


class Verifier:
    """Queue of verification waves, each a separate CPU process (engine python) so the GPU search never waits for it."""
    def __init__(self, out, workers, log):
        self.out, self.workers, self.log, self.q, self.proc, self.results, self.n = out, workers, log, [], None, [], 0
        for f in sorted(glob.glob(os.path.join(out, "verify_*.json"))):                                  # finished waves from an earlier (interrupted) run are reused, never repeated
            if not f.endswith("_tasks.json"): self.results += json.load(open(f)); self.n += 1
        if self.results: log("    resumed " + str(len(self.results)) + " finished verification checks: " + self.summ(self.results))

    def submit(self, label, tasks):
        self.n += 1; self.q.append((label, tasks)); self.poll()

    def poll(self):
        if self.proc and self.proc[0].poll() is not None:
            p, outp, label, t0 = self.proc; self.proc = None
            if p.returncode != 0: raise RuntimeError(f"verification wave {label} failed (see {self.out}/verify_{label}.log)")
            r = json.load(open(outp)); self.results += r; self.log(f"    [verify {label}] finished in {time.time() - t0:.0f}s: " + self.summ(r))
        if not self.proc and self.q:
            label, tasks = self.q.pop(0); tp = os.path.join(self.out, f"verify_{label}_tasks.json"); outp = os.path.join(self.out, f"verify_{label}.json")
            json.dump(tasks, open(tp, "w")); env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUTF8="1"); env.pop("DEFENGINE_GPU", None)
            lf = open(os.path.join(self.out, f"verify_{label}.log"), "w")
            self.proc = (subprocess.Popen([str(ENGINE_PY), "-m", "defengine.verify_batch", tp, outp, str(self.workers)], cwd=str(ROOT), env=env, stdout=lf, stderr=lf), outp, label, time.time())
            self.log(f"    [verify {label}] started on the CPU: {len(tasks)} checks, {self.workers} at a time")

    def wait(self):
        while self.proc or self.q: time.sleep(2); self.poll()

    def tested(self):
        t = {x["cement"] for x in self.results} | {x["cement"] for _, ts in self.q for x in ts}
        if self.proc: t |= {x["cement"] for x in json.load(open(self.proc[1].replace(".json", "_tasks.json")))}
        return t

    @staticmethod
    def summ(r):
        by = {}
        for x in r: by.setdefault(x["cement"], []).append(x["PASS"])
        return ", ".join(f"{c}:{sum(v)}/{len(v)} pass" for c, v in sorted(by.items()))


class Frontier:
    def __init__(self, a, log):
        from . import corelib as C, gpu_solver as GS, fast_solver as FS
        import torch
        self.C, self.GS, self.FS, self.torch, self.a, self.log = C, GS, FS, torch, a, log
        C.engine(); eng = GS._state()["eng"]; self.dev = eng.dev; eng.set_lowprec(not a.exact_precision); self.tt = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=self.dev)
        self.grid = list(range(a.lo, a.hi + 1, a.step)); self.G = len(self.grid)
        self.screen, self.fresh = self.mkbank(256, 900_000_001), self.mkbank(4096, 900_000_501)
        self.Z, self.S, self.res = np.zeros((0, 11)), np.zeros((0, self.G)), {}; self.t0 = time.time(); self.stage_t = {}; self.events = []

    def mkbank(self, n, s):
        es, eq, _ = self.C.res_bank_iid(n, seed=s + 2); return self.C.mfg_bank(n, seed=s + 1), self.tt(es), self.tt(eq), self.tt(np.random.default_rng(s + 3).random(n))

    def score(self, Z, c, bk):
        """Reliability on a bank; a design with ANY unsupported scenario cannot pass verification, so it scores below zero."""
        out = []
        for i in range(0, len(Z), 256):
            _, _, rh, _, sf = self.GS.objective(self.tt(Z[i:i + 256]), c, *bk); out.append(self.torch.where(sf >= 1 - 1e-12, rh, rh - 1.0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def add(self, zs, full=True):
        """Add designs to the archive (skip near-duplicates); optionally score the new ones at every grid level on the fresh bank. Returns their archive indices."""
        new = []
        for z in zs:
            z = np.asarray(z, float)
            if len(z) == 11 and (len(self.Z) == 0 or np.abs(self.Z - z).max(1).min() > 0.003): self.Z = np.vstack([self.Z, z]); new.append(len(self.Z) - 1)
        S = np.full((len(new), self.G), np.nan)
        if full and new:
            for gi, c in enumerate(self.grid): S[:, gi] = self.score(self.Z[new], c, self.fresh)
        self.S = np.vstack([self.S, S]); return new

    def load_prior(self, dirs, K=40):
        zs = []
        for d in dirs:
            for f in ([d] if d.endswith(".json") else sorted(glob.glob(os.path.join(d, "level_*.json")))):
                j = json.load(open(f)); zs += j["Z"] if "Z" in j else [b["z"] for b in j["best"][:6]]          # a level file, or an archive.json from a previous run
        new = self.add(zs, full=False)
        for gi, c in enumerate(self.grid):                           # cheap screen of everything, then the fresh bank only for the most promising K per level
            sc = self.score(self.Z[new], c, self.screen); top = np.array(new)[np.argsort(-sc)[:K]]; self.S[top, gi] = self.score(self.Z[top], c, self.fresh)
        self.log(f"[0] archive: {len(zs)} designs from earlier runs, {len(new)} distinct")

    def gv(self, gi):
        col = self.S[:, gi]; return float(np.nanmax(col)) if len(col) and not np.all(np.isnan(col)) else 0.0

    def v(self, c):
        if c in self.res: return self.res[c]["v"]
        return self.gv(self.grid.index(c)) if c in self.grid else -1.0

    def pool_for(self, c, n=40):
        """Best n archive designs AT THIS EXACT LEVEL (small bank, whole archive, ~1 s). Ranking by the nearest grid levels instead lets designs that are great at 275 crowd out designs that are best at 255."""
        if len(self.Z) < 5: return None
        return self.Z[np.argsort(-self.score(self.Z, c, self.screen))[:n]].tolist()

    def process_level(self, c, tag="a", gens=None):
        a = self.a; seed = a.seed + 10 * c + sum(map(ord, tag)); fn = os.path.join(a.out, f"level_{c}_{tag}.json"); pool = self.pool_for(c)
        if os.path.exists(fn): sol = json.load(open(fn))
        else:
            sol = self.FS.solve_level_fast(c, seed=seed, pool=pool, max_gens=gens or (a.max_gens if pool else int(1.5 * a.max_gens)), P=a.P)
            json.dump(sol, open(fn, "w"), indent=1)
        cands = [b["z"] for b in sol["best"]]; self.add(cands)
        U = np.array(cands + (pool if pool else [])); sc = self.score(U, c, self.fresh); o = np.argsort(-sc)[:5]
        v = float(sc[o[0]])
        if c in self.grid: v = max(v, self.gv(self.grid.index(c)))
        old = self.res.get(c); top = [dict(score=float(sc[i]), z=U[i].tolist()) for i in o]
        if old and old["v"] > v: top, v = old["top"], old["v"]
        self.events.append(dict(wall_s=round(time.time() - self.t0, 1), what="level", cement=c, tag=tag, solve_s=round(sol["seconds"], 1), reliability=round(v, 4))); json.dump(self.events, open(os.path.join(a.out, "timing.json"), "w"), indent=1)
        self.res[c] = dict(v=v, top=top, seconds=sol["seconds"], gens=sol["generations"]); self.log(f"    level {c} [{tag}]: reliability {v:.3f} ({sol['seconds']:.0f}s, {sol['generations']} generations)")
        return v

    def tasks(self, levels):
        out = []
        for c in levels:
            for k, d in enumerate(self.res[c]["top"][:2]):
                nom = self.GS.decode(self.tt([d["z"]]), c); nominal = dict(C=float(c), r=float(nom["r"][0]), A=float(nom["A"][0]), P=float(nom["P"][0]), S=float(nom["S"][0]), phi=float(nom["phi"][0]), g=nom["g"][0].cpu().numpy().tolist())
                out += [dict(cid=f"FF_C{c}_d{k}", nominal=nominal, replicate=r, cement=c) for r in range(2)]
        return out

    def run(self):
        a, log, tgt = self.a, self.log, self.a.target; os.makedirs(a.out, exist_ok=True)
        self.load_prior(list(a.prior) + [a.out])
        gv = {c: self.gv(i) for i, c in enumerate(self.grid)}; log("[0] instant frontier from the archive alone: " + ", ".join(f"{c}:{x:.2f}" for c, x in gv.items()))
        self.stage_t["archive"] = time.time() - self.t0; ver = Verifier(a.out, a.verify_workers, log); t1 = time.time()
        solved = lambda: sorted(self.res); hit = lambda c: self.v(c) >= tgt
        # 1 - walk down
        good = [c for c in self.grid if gv[c] >= tgt - 0.02]; start = min(good) if good else max(gv, key=gv.get)
        log(f"[1] walk: start at {start} kg/m3, then down in steps of {a.step}")
        c = start; self.process_level(c)
        while not hit(c) and not any(hit(x) for x in solved()) and c + a.step <= a.hi: c += a.step; self.process_level(c)        # nothing passes yet: climb until something does
        c = min(x for x in solved() if hit(x)) - a.step if any(hit(x) for x in solved()) else None; miss = 0
        while c is not None and c >= a.lo and miss < 2:
            self.process_level(c); miss = 0 if hit(c) else miss + 1; c -= a.step
        hits = [x for x in solved() if hit(x)]; self.stage_t["walk"] = time.time() - t1; t1 = time.time()
        if not hits: log("[1] nothing reached the target in this cement range"); return self.finish(ver)
        b = min(hits); below = [x for x in solved() if x < b and not hit(x)]; lo_miss = max(below) if below else None
        def vq(levels):
            lv = [x for x in levels if x not in ver.tested()]
            if lv and not a.no_verify: ver.submit(f"w{ver.n + 1}", self.tasks(lv))
        vq(sorted(hits)[:2])                                                                 # wave 1 runs on the CPU while the refine stage keeps the GPU busy
        # 2 - refine
        if lo_miss is not None:
            log(f"[2] refine between {lo_miss} (miss) and {b} (hit): 5 kg, then 1 kg")
            for x in range(lo_miss + 5, b, 5): self.process_level(x); ver.poll()
            p = min(x for x in solved() if hit(x))
            for x in range(max(lo_miss + 1, p - 4), p): self.process_level(x); ver.poll()
        else: log(f"[2] the target is still met at the bottom of the range ({a.lo}); extend --lo to find the minimum")
        low = sorted(x for x in solved() if hit(x))[:3]
        for x in low: self.process_level(x, tag="b", gens=a.max_gens); ver.poll()                        # independent second search at the lowest hits
        self.stage_t["refine"] = time.time() - t1; t1 = time.time()
        vq([c for c in sorted(self.res) if self.res[c]["v"] >= tgt - 0.02][:3])             # the lowest crossing candidates are checked first
        if a.curve:                                                                          # fill in the rest of the curve: every grid level, verified in batches of 3
            rest = [c for c in self.grid if c not in self.res]; log(f"[3] curve: {len(rest)} more levels {rest}")
            for i, c in enumerate(rest):
                self.process_level(c); ver.poll()
                if i % 3 == 2 or i == len(rest) - 1: vq([x for x in rest[max(0, i - 2):i + 1] if self.res[x]["v"] >= tgt - 0.25])
            self.stage_t["curve"] = time.time() - t1
        vq([c for c in self.grid if c in self.res and self.res[c]["v"] >= tgt - 0.25])      # every solved grid level on the curve gets checked (levels already checked are skipped)
        return self.finish(ver)

    def finish(self, ver):
        a, log = self.a, self.log; t1 = time.time(); tgt = a.target
        json.dump(dict(Z=self.Z.tolist()), open(os.path.join(a.out, "archive.json"), "w"))
        cand = sorted(c for c in self.res if self.res[c]["v"] >= tgt - 0.02)
        if not a.no_verify:
            ver.wait(); tested = {x["cement"] for x in ver.results}
            while True:                                      # waves of the lowest untested candidates until the lowest verified level is known
                ok = self.verified_levels(ver.results); untested = [c for c in cand if c not in tested]
                if not untested or (ok and not [c for c in untested if c < min(ok)]): break
                nxt = untested[:3]; ver.submit(f"w{ver.n + 1}", self.tasks(nxt)); ver.wait(); tested |= set(nxt)
        return self.report(ver, time.time() - t1)

    @staticmethod
    def verified_levels(results):
        by = {}
        for x in results: by.setdefault((x["cement"], x["candidate_id"]), []).append(x["PASS"])
        return sorted({c for (c, _), p in by.items() if len(p) >= 2 and all(p)})

    def report(self, ver, vsec):
        a, log = self.a, self.log; rows = []; ok = self.verified_levels(ver.results)
        for c in sorted(self.res):
            vs = [x for x in ver.results if x["cement"] == c]; best = max((x["R_LCB95_joint"] for x in vs), default=None)
            rows.append(dict(cement=c, search_reliability=round(self.res[c]["v"], 4), verified=bool(c in ok), verified_R_LCB95_best=best, n_checks=len(vs)))
        total = time.time() - self.t0
        out = dict(levels=rows, verified_minimum=(min(ok) if ok else None), verified_levels=ok, seconds=total, stage_seconds=self.stage_t | dict(verify_tail=vsec), args=vars(a), verification=ver.results)
        json.dump(out, open(os.path.join(a.out, "frontier.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        log("[result] level | search reliability | verified | best verified R_LCB95"); [log(f"   {r['cement']:4d} | {r['search_reliability']:.3f} | {r['verified']} | {r['verified_R_LCB95_best']}") for r in rows]
        log(f"[result] lowest cement level whose design passed both verification replicates: {min(ok) if ok else 'none'} kg/m3; total time {total / 60:.1f} min")
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--prior", nargs="*", default=[]);
    ap.add_argument("--lo", type=int, default=200); ap.add_argument("--hi", type=int, default=500); ap.add_argument("--step", type=int, default=25); ap.add_argument("--target", type=float, default=0.935)
    ap.add_argument("--max-gens", type=int, default=100); ap.add_argument("--P", type=int, default=100); ap.add_argument("--seed", type=int, default=2000); ap.add_argument("--verify-workers", type=int, default=4)
    ap.add_argument("--no-verify", action="store_true"); ap.add_argument("--curve", action="store_true", help="solve and verify every grid level, not only the crossing"); ap.add_argument("--exact-precision", action="store_true", help="search in double precision (slower; equivalence tested, see E-08)"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, str(ROOT)); from defengine import corelib as C
    bad = C.verify_core(); assert not bad, f"frozen core changed: {bad[:3]}"
    lf = open(os.path.join(a.out, "run.log"), "a", encoding="utf-8")
    T0 = time.time()
    def log(m): m = "[" + format((time.time() - T0) / 60, "5.1f") + " min] " + m; print(m, flush=True); lf.write(m + chr(10)); lf.flush()
    log(f"fast frontier | model hash {C.model_hash()[:16]} | targets {C.S_STAR} MPa / {C.K_STAR} mm/s / {C.REL_TARGET} | search target {a.target}")
    Frontier(a, log).run(); log("DONE")
