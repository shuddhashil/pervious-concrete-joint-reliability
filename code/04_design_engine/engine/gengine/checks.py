"""Selection and tiers from the walk logs only (plan section 8) - no GPU. Evidence thresholds n_min 1/2/3 and nominal-only vs 2 percent acceptance all come from the same log."""
import json, math


def load_log(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def attempts(lines):
    return [l for l in lines if l.get("event") == "attempt"]


def accepted(lines):
    return [l for l in attempts(lines) if l["accepted"]]


def data_ok(l, n_min):
    ev = l["accept"]["evidence"]; return ev["n_near_pass"] >= n_min and not ev["conflict"]["flag"]


def lowest(lines, pred=lambda l: True):
    c = [l for l in accepted(lines) if pred(l)]; return min(c, key=lambda l: l["C"]) if c else None


def parent_summary(lines, n_mins=(1, 2, 3)):
    """model_min = lowest accepted cement; data_min[n] = lowest accepted cement with >= n near passing units and no conflict; nominal_only_min = lowest ATTEMPTED cement whose nominal LB cleared the walk threshold."""
    at = attempts(lines); R = at[0]["accept"]["R_ACC"] if at else None
    nom_ok = [l for l in at if l["accept"]["nominal"]["LB"] >= R and l["accept"]["nominal"]["support_fraction"] >= 1 - 1e-12]
    out = dict(model_min=(lowest(lines) or {}).get("C"), data_min={n: (lowest(lines, lambda l, n=n: data_ok(l, n)) or {}).get("C") for n in n_mins},
               nominal_only_min=min((l["C"] for l in nom_ok), default=None), n_attempts=len(at), n_accepted=len(accepted(lines)))
    return out


def nesting_violations(summ):
    """Tighter requirement can never give lower cement: data_min(1) <= data_min(2) <= data_min(3) <= ... and model_min <= data_min(1); nominal_only_min <= model_min (None = infinity)."""
    inf = math.inf; v = lambda x: inf if x is None else x; bad = []
    seq = [("nominal_only_min", v(summ["nominal_only_min"])), ("model_min", v(summ["model_min"]))] + [(f"data_min[{n}]", v(summ["data_min"][n])) for n in sorted(summ["data_min"])]
    for (an, a), (bn, b) in zip(seq, seq[1:]):
        if a > b + 1e-9: bad.append(f"{an}={a} > {bn}={b}")
    return bad


def tier(family_has_recorded_passer, data_min_exists, certificate_ok):
    if not family_has_recorded_passer: return "Tier 2 (model-suggested, outside the data, needs lab trial; never recommended)"
    if data_min_exists and certificate_ok: return "Tier 1"
    return "not reachable (no data-backed design)" if not data_min_exists else "Tier 1 candidate (certificate pending/failed)"
