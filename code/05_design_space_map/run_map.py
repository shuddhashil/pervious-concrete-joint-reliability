"""Driver for the v3 design-space map. Runs the pipeline as subprocesses with cwd = ROOT and sys.executable, in order, skipping a step whose
marker logs_v3/done_<step>.txt exists and writing the marker only when the step returns 0; stops on the first failure. Streams every step's output to logs_v3/<step>.log and prints
one line per step with the elapsed minutes. Before the steps sets GENGINE_BASE / GENGINE_CONFIG (the 18/13 target, as sm_common.py does) and PYTHONDONTWRITEBYTECODE=1.
After the last step zips supportmap_v3 (everything except inputs/ and old_code/) to /content/results/map_v3_results.zip when /content exists, else supportmap_v3/map_v3_results.zip.
usage: python run_map.py   (env MAP_REFINE_MIN = refinement budget in minutes, default 60)"""
import sys, os, time, zipfile, subprocess
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM); RUNS = os.path.join(ROOT, "runs")
L3 = os.path.join(SM, "logs_v3") if os.path.isdir("/content") else os.path.join(RUNS, "logs_v3")
STEPS = ["prep_target", "harvest_six", "rescore", "classify0", "run_refine", "run_final_prep", "run_certs_v2", "fix_monotone", "islands", "assemble_v3", "make_figure_v3"]


def zip_results():
    """zip supportmap_v3 except inputs/ and old_code/ (the archive itself is never added)."""
    dest = os.path.join("/content", "results", "map_v3_results.zip") if os.path.isdir("/content") else os.path.join(RUNS, "map_v3_results.zip")
    os.makedirs(os.path.dirname(dest), exist_ok=True); n = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(SM):
            if os.path.abspath(root) == os.path.abspath(SM): dirs[:] = [d for d in dirs if d not in ("inputs", "old_code")]
            for fn in files:
                p = os.path.join(root, fn)
                if os.path.abspath(p) == os.path.abspath(dest): continue
                z.write(p, os.path.relpath(p, SM)); n += 1
    print(f"zip: {dest} ({n} files)"); return dest


def main():
    os.makedirs(L3, exist_ok=True); TGT = os.path.join(ROOT, "targets", "S18_K13_R0.90")
    env = os.environ.copy(); env["GENGINE_BASE"] = TGT; env["GENGINE_CONFIG"] = os.path.join(TGT, "CONFIG.json"); env["PYTHONDONTWRITEBYTECODE"] = "1"
    for s in STEPS:
        mark = os.path.join(L3, f"done_{s}.txt"); log = os.path.join(L3, f"{s}.log")
        if os.path.exists(mark): print(f"{s}: skipped (marker exists)"); continue
        cmd = [sys.executable, os.path.join(SM, "code", s + ".py")]
        if s == "run_refine": cmd.append(os.environ.get("MAP_REFINE_MIN", "60"))
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60.0
        if rc != 0: print(f"{s}: FAILED rc={rc} after {dt:.2f} min (see {log})"); return rc
        open(mark, "w").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} rc=0 {dt:.2f} min\n")
        print(f"{s}: ok, {dt:.2f} min")
    zip_results(); print("MAP DONE"); return 0


if __name__ == "__main__":
    sys.exit(main())
