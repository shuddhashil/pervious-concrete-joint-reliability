"""Verification worker (acceptance authority, CPU, engine python): python -m defengine.verify_batch <tasks.json> <out.json> <workers>
Each task: dict(cid, nominal, replicate, cement). Runs the FROZEN estimator via verify.verify_design with disjoint seeds. Launched by fast_frontier as a separate process
so verification runs on the CPU while the GPU keeps searching."""
import sys, json
from concurrent.futures import ProcessPoolExecutor


def _init():
    from . import corelib as C
    C.engine()


def _one(t):
    from . import verify as V
    return V.verify_design(t["cid"], t["nominal"], replicate=t["replicate"]) | dict(cement=t["cement"], stage="verify")


if __name__ == "__main__":
    tasks = json.load(open(sys.argv[1])); n = int(sys.argv[3])
    with ProcessPoolExecutor(max_workers=n, initializer=_init) as p:
        res = list(p.map(_one, tasks))
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
