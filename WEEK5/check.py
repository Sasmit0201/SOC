"""check.py — validate your rule and see a practice score before you submit.

    python check.py            # checks my_rule.npz
    python check.py other.npz
"""

import sys

import numpy as np

import truth as T

MAX_PARAMS = 100_000


def validate(path):
    p = path if path.endswith(".npz") else path + ".npz"
    try:
        d = np.load(p, allow_pickle=False)
    except Exception as e:
        return False, f"cannot read as a safe .npz ({type(e).__name__})"
    if not {"sizes", "flat"} <= set(d.files):
        return False, "not a saved rule (missing 'sizes'/'flat')"
    if ("kind" not in d.files) or str(d["kind"]) != "truth-nca":
        return False, "not a Truth Arena rule"
    sizes = [int(x) for x in d["sizes"]]
    if sizes != T.RULE_SIZES:
        return False, f"shape {sizes} but the arena needs {T.RULE_SIZES}"
    flat = np.asarray(d["flat"])
    exp = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    if flat.shape != (exp,):
        return False, f"weight count {flat.shape} != expected ({exp},)"
    if flat.size > MAX_PARAMS:
        return False, f"too big ({flat.size} weights)"
    if not np.all(np.isfinite(flat)):
        return False, "weights contain NaN/Inf"
    note = str(d["note"]) if "note" in d.files else ""
    if len(note.strip()) < 40 or "TODO" in note.upper():
        return False, ("design note missing/too short — set NOTE in solution.py to "
                       "1-2 real sentences on WHY your society survives the liars")
    return True, "ok"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "my_rule.npz"
    ok, info = validate(path)
    if not ok:
        print(f"FAIL  {path}\n  -> {info}\nFix this and re-run before submitting.")
        sys.exit(1)
    print(f"PASS  {path}  (a valid Truth Arena rule)")
    r = T.TruthRule.load(path)
    d = T.evaluate(r, seeds=(0, 1, 2, 3, 4))
    print(f"  practice score : {d['score']:.3f}")
    print(f"  society size   : {d['size']:.0f} cells")
    print(f"  cohesion       : {d['cohesion']:.2f}  (higher = more organized)")
    print("Looks valid — safe to submit. (The real arena is harder & hidden — build "
          "something robust, not tuned to practice.)")


if __name__ == "__main__":
    main()
