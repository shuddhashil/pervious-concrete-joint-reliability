"""Pool layer 2: space-filling (scrambled Sobol, fixed seeds) samples in a small box AROUND EVERY recorded de-duplicated recipe (all units, none chosen by any target), added because the whole-box Sobol pool of layer 1 did not
contain designs near the narrow region where the certified 18/13 and 20/10 designs lie (stage B at the cell of 18 MPa / 12.7 mm/s: best pool design 0.75, certified design 0.9252). Per unit: two Sobol sets of 32 points, half-widths 1x and 2.5x of
[cement +-50 kg/m3, w/c / ln CA / SP / sand +-0.10 of their range, porosity +-2 pp (kept in the 20-30 % window), six-bin blend +-0.05 per bin then renormalised]; blend projected into the unit's family; same screen as layer 1 (gate, family rule, support at the nominal blend). No search, no objective."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_common import *
from scipy.stats import qmc
N_PER = 32; SCALES = (1.0, 2.5)


def build(ctx, tg, log=print):
    d = ctx.data; lab = np.array(ctx.cfg["families"]["labels"]); ctx.eng.set_lowprec(False); bank, u = ctx.bank(256, 20261004); lo = lambda k: BND[k][0]; sp = lambda k: BND[k][1] - BND[k][0]; Z = []; Cv = []; fam = []; unit = []
    for uu in range(d.nU):
        R = d.Ru[uu]; g0 = d.Gu[uu]; f = int(lab[d.blend_of[uu]])
        for si, sc in enumerate(SCALES):
            U = qmc.Sobol(12, scramble=True, seed=7000 + 10 * uu + si).random(N_PER) * 2 - 1
            c = np.clip(R[0] + 50.0 * sc * U[:, 0], 200.0, BND["C"][1]); z = np.zeros((N_PER, 11)); z[:, 0] = np.clip((R[1] - lo("r")) / sp("r") + 0.10 * sc * U[:, 1], 0, 1)
            z[:, 1] = np.clip((math.log(max(R[2], 1e-9)) - C.SC.LNA_LO) / (C.SC.LNA_HI - C.SC.LNA_LO) + 0.10 * sc * U[:, 2], 0, 1)
            z[:, 2] = np.clip((R[3] - lo("P")) / sp("P") + 0.10 * sc * U[:, 3], 0, 1); z[:, 3] = np.clip((R[4] - lo("S")) / sp("S") + 0.10 * sc * U[:, 4], 0, 1)
            ph = np.clip(np.clip(R[5], *PHI_WIN) + 2.0 * sc * U[:, 5], *PHI_WIN); z[:, 4] = (ph - lo("phi")) / sp("phi")
            g = np.clip(g0[None, :] + 0.05 * sc * U[:, 6:12], 0, None); g = g / g.sum(1, keepdims=True); z[:, 5:] = g; Z.append(z); Cv.append(c); fam.append(np.full(N_PER, f)); unit.append(np.full(N_PER, uu))
    Z = np.concatenate(Z); Cv = np.concatenate(Cv); fam = np.concatenate(fam); unit = np.concatenate(unit); keep = []
    with torch.no_grad():
        for f in (0, 1, 2):
            ix = np.flatnonzero(fam == f); Zt = T(Z[ix]); g2 = ctx.fam.project(decode(Zt, T(Cv[ix]))["g"], f); Zt = torch.cat([Zt[:, :5], g2], 1); Z[ix] = Zt.cpu().numpy(); nom = decode(Zt, T(Cv[ix]))
            pen = torch.nan_to_num(ctx.gate.penalties(nom)["penalty"], posinf=1e6); fok, fd, _ = ctx.fam.check(nom["g"], f); ok = ((pen <= 0) & fok).nonzero()[:, 0]; kk = []
            for a in range(0, len(ok), 1500):
                tg(); ii = ok[a:a + 1500]; st = ctx.eng.stats_t({k: v[ii] for k, v in nom.items()}, bank, None, None, u=u); kk.append(ii[st["support_fraction"] >= 1 - 1e-12])
            kk = torch.cat(kk).cpu().numpy() if kk else np.zeros(0, int); keep.append(ix[kk]); log(dict(fam=f, raw=len(ix), gate_family_ok=int(len(ok)), kept=int(len(kk))))
    keep = np.sort(np.concatenate(keep)); return dict(Z=Z[keep], C=Cv[keep], fam=fam[keep], stratum=np.full(len(keep), 3), typ=np.full(len(keep), 2), unit=unit[keep])


if __name__ == "__main__":
    t0 = time.time(); cfg, H, ctx = get_ctx(); tg = TempGuard(); pool = build(ctx, tg); np.savez(os.path.join(SM, "data", "pool_anchor_raw.npz"), **pool)
    json.dump(dict(seconds=round(time.time() - t0, 1), n_kept=int(len(pool["C"])), n_units=int(ctx.data.nU), gpu_peak=tg.peak, pauses=tg.pauses), open(os.path.join(SM, "logs", "pool_anchor_build.json"), "w"), indent=1); print("anchored pool", len(pool["C"]), round(time.time() - t0), "s")
