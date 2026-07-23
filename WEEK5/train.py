"""Train a Truth Arena rule — search for weights that grow a society which
survives the liars.

    python train.py --generations 40 --out my_rule --gif

No gradients: a genetic algorithm. A good rule grows a fed, DENSE society and
defends its frontier — it does NOT sprawl everywhere (that starves and gets
swept away by misinformation).
"""

import argparse

import numpy as np

import truth as T


def evolve(generations=40, pop=18, elite=5, mutation=0.12, seeds=(0, 1, 2),
           seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    tpl = T.TruthRule(seed=seed)
    dim = tpl.n_params
    population = [rng.standard_normal(dim).astype(np.float32) * 0.3 for _ in range(pop)]
    best_v, best_f = None, -1.0
    for g in range(generations):
        fits = [T.evaluate(tpl.set_flat(v), seeds=seeds)["score"] for v in population]
        order = np.argsort(fits)[::-1]
        population = [population[i] for i in order]
        fits = [fits[i] for i in order]
        if fits[0] > best_f:
            best_f, best_v = fits[0], population[0].copy()
        if verbose:
            print(f"gen {g:2d}   best {fits[0]:.3f}   mean {np.mean(fits):.3f}"
                  f"   (all-time {best_f:.3f})")
        parents = population[:elite]
        population = parents + [
            parents[int(rng.integers(0, elite))]
            + rng.standard_normal(dim).astype(np.float32) * mutation
            for _ in range(pop - elite)]
    return T.TruthRule().set_flat(best_v), best_f


def main():
    p = argparse.ArgumentParser(description="Evolve a Truth Arena rule")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--pop", type=int, default=18)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--mutation", type=float, default=0.12)
    p.add_argument("--seeds", type=int, default=4,
                   help="practice worlds per candidate (more = more robust, slower)")
    p.add_argument("--out", default="my_rule")
    p.add_argument("--gif", action="store_true", help="save a replay of the best rule")
    p.add_argument("--note", default="", help="your design note (required to submit)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"evolving a Truth Arena rule ({args.generations} generations)…")
    rule, fit = evolve(generations=args.generations, pop=args.pop, elite=args.elite,
                       mutation=args.mutation, seeds=tuple(range(args.seeds)),
                       seed=args.seed)
    rule.save(args.out, note=args.note)
    print(f"\nbest practice score {fit:.3f}  ->  saved {args.out}.npz")
    if len(args.note.strip()) < 40:
        print("NOTE: add a design note before submitting  (--note \"why it survives\")")
    if args.gif:
        import render
        frames = T.run(rule, seed=0, record=True)["frames"]
        out = render.save_gif(frames, args.out + ".gif", title="my society")
        print(f"replay -> {out}")
    print("Check it before submitting:  python check.py " + args.out + ".npz")


if __name__ == "__main__":
    main()
