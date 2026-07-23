"""
============================================================================
  YOUR WORKSPACE  —  Truth Arena
============================================================================
This is the ONE file you work in. The whole world is already imported for you.
You edit the two sections marked  >>> YOUR CODE <<<  and then run:

      python solution.py

It trains your rule and saves  my_rule.npz  — your submission. After it runs:

      python check.py my_rule.npz      (make sure it says PASS)

You don't need to touch any other file. If you're curious how the world works,
open truth.py — but you never edit it.
----------------------------------------------------------------------------
"""

import numpy as np

# the world, imported for you. These are the only things you need:
from truth import TruthRule, run
#   TruthRule()          -> a fresh (untrained) rule
#   rule.n_params        -> how many weights the rule has (~600)
#   rule.get_flat()      -> the weights, as one flat array of numbers
#   rule.set_flat(w)     -> load a flat array of numbers as the rule's weights
#   run(rule, seed=0)    -> play one society; returns a dict with:
#         result["size"]      how many TRUTH cells are alive at the end
#         result["cohesion"]  how clumped-together they are (0..1)
#         result["score"]     the official score = size-fraction x cohesion


# >>> REQUIRED before submitting: replace this with 1-2 sentences on WHY your
#     society survives the liars. A submission without a real note is rejected. <<<
NOTE = "The evolved rule encourages dense, well-connected colonies that balance growth with cohesion. This limits infection cascades while allowing the society to expand and remain resilient against repeated liar attacks."


# ===========================================================================
#  SECTION 1  >>> YOUR CODE <<<   —  what makes a GOOD society?
#
#  This is your THEORY of the game. While training, we reward each rule by
#  this number. The official score is size x cohesion — but here you can guide
#  learning however you believe wins. Try different formulas!
# ===========================================================================
def how_good_is(rule):
    scores = []

    for seed in range(12):
        r = run(rule, seed=seed)

        fitness = (
            6.0 * r["score"] +
            2.0 * r["cohesion"] +
            0.003 * r["size"]
        )

        scores.append(fitness)

    scores = np.asarray(scores)

    return (
        0.60 * scores.mean() +
        0.25 * np.median(scores) +
        0.15 * scores.min()
    )

# ===========================================================================
#  SECTION 2  >>> YOUR CODE <<<   —  how do you SEARCH for good weights?
#
#  A rule is just ~600 numbers. This is a simple evolution: keep a group of
#  rules, score them, keep the best, make mutated copies, repeat. Tune the
#  knobs, or rewrite the loop with your own idea.
# ===========================================================================
POP         =32      # how many rules in the group
GENERATIONS = 40      # how many rounds of improvement (higher = better, slower)
KEEP        = 8      # how many top rules survive each round
MUTATION = 0.12 # how much to jiggle the weights when copying


def tournament(scored, rng, k=3):
    """Select one parent using tournament selection."""
    idx = rng.choice(len(scored), k, replace=False)
    idx.sort()                     # scored is already sorted (best first)
    return scored[idx[0]][1]


def crossover(p1, p2, rng):
    """Uniform crossover."""
    mask = rng.random(len(p1)) < 0.5
    return np.where(mask, p1, p2).astype(np.float32)


def mutate(child, rng, sigma):
    """Sparse Gaussian mutation."""
    child = child.copy()

    mask = rng.random(len(child)) < 0.08      # mutate 8% of weights

    child[mask] += (
        rng.standard_normal(mask.sum()).astype(np.float32)
        * sigma
    )

    return child


def find_best_rule():
    rng = np.random.default_rng(0)
    n = TruthRule().n_params

    group = [
        rng.standard_normal(n).astype(np.float32) * 0.3
        for _ in range(POP)
    ]

    best_weights = None
    best_score = -np.inf

    for gen in range(GENERATIONS):

        scored = []

        for weights in group:
            rule = TruthRule().set_flat(weights)
            fitness = how_good_is(rule)
            scored.append((fitness, weights))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_weights = scored[0][1].copy()

        print(
            f"Generation {gen:3d} | "
            f"Best {scored[0][0]:.4f} | "
            f"Overall {best_score:.4f}"
        )

        elites = [w.copy() for _, w in scored[:KEEP]]

        sigma = max(
            0.02,
            MUTATION * (1 - gen / GENERATIONS)
        )

        new_group = elites

        while len(new_group) < POP:

            # 10% random immigrants
            if rng.random() < 0.10:
                child = (
                    rng.standard_normal(n).astype(np.float32)
                    * 0.3
                )

            else:
                p1 = tournament(scored, rng)
                p2 = tournament(scored, rng)

                child = crossover(p1, p2, rng)
                child = mutate(child, rng, sigma)

            new_group.append(child)

        group = new_group

    return TruthRule().set_flat(best_weights)

# ===========================================================================
#  Run it. (You usually don't need to change anything below.)
# ===========================================================================
if __name__ == "__main__":
    print("training your rule…")
    rule = find_best_rule()
    rule.save("my_rule", note=NOTE)
    print("\nsaved my_rule.npz")
    print("now run:  python check.py my_rule.npz")
