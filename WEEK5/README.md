# 🧠 Truth Arena — Build a Society That Beats the Lies

Welcome to the final challenge. You've done Conway's Game of Life, Neural Cellular
Automata, and Boids. Now you put them together into one thing.

## The challenge

You are given a grid. On it, a small society of **truth** cells tries to grow.
There is **food** to grow on, and there are **liars** — misinformation that
spreads by touching your cells and flipping them. A liar cell keeps spreading.

You **do not** control the world. You design **one rule** — a tiny brain that
every truth cell runs, at the same time, every step. From that single local rule,
a whole society has to **emerge**: grow big, stay together, feed itself, and
survive the liars.

> After **70 steps**, we measure your society. The **bigger and more organized**
> it is (and still standing), the higher your score. Best society wins.

## The trap (read this twice)

The obvious move — **grow everywhere as fast as possible** — is a trap.

- A society that sprawls out gets **thin and hungry** far from food. Weak, starving
  cells are **easy to convert**. One liar gets in, flips a weak cell, that becomes
  a liar, which flips the next... and the whole thing **cascades** and dies.
- A society that just **huddles** in one safe spot stays immune — but it stays
  **tiny**, and tiny scores low.

So the real game is a balance:

```
   grab FOOD  →  you must spread out  ─┐
                                        ├──  you can't do both the easy way.
   stay IMMUNE →  you must stay dense  ─┘      Being clever is the point.
```

**Immunity comes from being together.** A truth cell surrounded by enough truth
neighbours resists liars. A lone or starving cell doesn't. Safety in numbers.

## Your job: design the brain

Your submission is the weights of a tiny neural net (~600 numbers). Every truth
cell runs it. It reads what's around the cell and decides what to do.

**What a cell sees** (a small window around it):

| it senses | meaning |
|---|---|
| `truth`  | friendly cells nearby (your density → your immunity) |
| `liar`   | the threat nearby |
| `food`   | where you can grow |
| `energy` | how well-fed this cell is |
| `memory` | 3 hidden numbers the cell can set for itself — use them to **signal** |

**What a cell outputs:** a **grow signal** (should I try to spread?) + updates to
its 3 **memory** numbers. That's it. Everything smart — walls, alarms, healing —
has to *emerge* from this one rule. There's no boss telling cells what to do.

## What's in this folder

| file | what it's for |
|---|---|
| **`solution.py`** | **your workspace — edit this**: the reward, the search, and your `NOTE` |
| `truth.py` | the world (read-only sandbox) — shows exactly what a cell senses |
| `check.py` | validate your `my_rule.npz` before you submit |
| `train.py` | a no-code quick trainer (alternative to `solution.py`) |
| `render.py` | draws the replay gifs (used by the others; you don't edit it) |
| `solution_nn.py` | *bonus:* train a neural net from scratch (forward + backprop, REINFORCE) |
| `grow_nca.py` | *bonus:* watch a from-scratch NCA grow a shape — gradients that really work |
| `requirements.txt` | the two libraries to install |

## How to play

You work in **one file: `solution.py`.** Open it — the whole world is already
imported. You edit just two sections marked `>>> YOUR CODE <<<`:

1. **what makes a good society** — your *reward*. This is your theory of the game.
   Reward big? dense? both? You decide what training aims for.
2. **how to search for good weights** — the training loop (a simple evolution you
   can tune or rewrite).

```bash
pip install -r requirements.txt   # once
python solution.py                # trains your rule -> my_rule.npz
python check.py my_rule.npz       # must say PASS
# submit my_rule.npz
```

That's it — you never edit any other file. Everything you can touch (`TruthRule`,
`run`, the weights) is imported at the top of `solution.py` with comments.

> Just want a quick button with no coding? `python train.py --generations 40
> --out my_rule --gif` runs a ready-made trainer and saves a replay.

Training is a **genetic algorithm**: keep a group of rules, score each by running
the society, keep the best, mutate, repeat. More rounds = a better, more robust rule.

## Commands cheat-sheet

| what you want | command |
|---|---|
| install (once) | `pip install -r requirements.txt` |
| **train your rule** | `python solution.py` |
| watch a replay of the best rule | `python train.py --out my_rule --gif` → open `my_rule.gif` |
| check a submission is valid | `python check.py my_rule.npz` |

Notes: you need **Python 3.9+**. On Mac/Linux use `python3` / `pip3` if `python`
doesn't work. If `pip install -r requirements.txt` fails, try
`pip install numpy matplotlib`.

## Do I need TensorFlow or PyTorch? — No 🙂

Some of you trained your Week-4 models in **TensorFlow**, some in **PyTorch**.
For this challenge you need **neither** — and here's the real reason:

- Your rule is a **tiny network already written for you in plain numpy** (inside
  `truth.py`). You don't build a model — you just find good **weights** for it.
- There are **no labels** to learn from. Your score comes from *running the
  society* (a simulation), so you can't do gradient descent / backprop like a DQN.
  Instead you **search** for weights (evolution) — and that works the same no
  matter which framework you came from.

So whichever you used before, everyone works in the same `solution.py` with just
numpy. One less thing to install, and it runs on any laptop.

> **Advanced (optional):** a rule is just `TruthRule().n_params` numbers. If you
> really want to use your own framework to shape them, you can — build the
> `21 → 24 → 4` net (see `truth.py`), flatten its weights in `[W0, b0, W1, b1]`
> order, then `TruthRule().set_flat(your_numbers).save("my_rule")`. Most people
> should just tune `solution.py`.

## Want to train a NEURAL NET yourself? (two bonus files)

The default (`solution.py`) *evolves* the weights. If you'd rather **build and
train a neural net from scratch** — no PyTorch, no TensorFlow, you write the
forward pass *and* the gradients yourself — there are two extra files:

- **`grow_nca.py`** 🌱 — *start here for the fun.* A tiny NCA that **grows from
  one seed into a target shape** (a heart). It's fully differentiable, so real
  gradient descent works: you'll see the **loss go down** and the picture appear.
  You hand-write the forward pass, backprop-through-time, and a mini-Adam.
  ```bash
  python grow_nca.py      # -> grow_nca.png + grow_nca.gif (watch it grow!)
  ```
  This is the clearest way to *feel* what backprop does — exactly Week 2.

- **`solution_nn.py`** 🧪 — the *ambitious* competition path. Same from-scratch
  net, trained with **REINFORCE** (policy-gradient RL, like a Week-4 DQN) on the
  real society. It's honest and educational, but RL here is high-variance, so it
  usually scores **lower** than evolution — treat it as "understand it deeply,"
  not "win fast."

For a strong submission, `solution.py` (evolution) is still the reliable path.

## Ideas to try (this is where you win)

There's no single right answer — many strategies work. Some to invent:

- **Membrane** 🧱 — grow a **dense outer wall** of truth cells that shields a core.
  The wall is immune (lots of neighbours); the inside grows safely.
- **Signalling** 📣 — use the hidden memory channels as an **alarm**: when a cell
  senses liars, it sets a memory value; neighbours read it and reinforce.
- **Smart foraging** 🍽️ — grow *toward* food, but never let your edge get so thin
  it drops below the immunity line.
- **Self-healing** 🩹 — when the liars tear a hole, sense the gap and **regrow** it
  before they pour through.

Mix them. Try weird things. Watch the gif and ask *why* it died — then fix that.

## How you're scored

```
   score = society size × cohesion   (at step 70)
             │             └── how clumped-together your cells are
             └── how many truth cells are alive
```

Big **and** organized **and** alive beats everything. Scattered survivors, a tiny
huddle, or a society that got overrun all score low.

## Two rules that decide whether you actually learn this

**1. A design note is required.** Before you submit, set `NOTE` at the top of
`solution.py` to **1–2 real sentences on *why* your society survives the liars**
(your strategy — membrane? signalling? foraging?). It's saved inside `my_rule.npz`.
`check.py` and the host both **reject a submission with no note** (or a "TODO").
You can't win by luck you can't explain.

**2. You're scored on your WORST difficulty.** The hidden test runs your rule at
**three liar-aggression tiers — easy / medium / hard — and your rank is your
weakest one.** A rule that only works when the liars are gentle will crack on
hard and rank low. Only a *robust* strategy scores well everywhere.

## Fair play

- Submit **one file**: `my_rule.npz` (weights + your note). Only data — no code runs.
- The **real test is hidden**: unseen worlds, tougher liars, three tiers. Don't
  tune to one practice map — build a rule that holds up **anywhere**.
- Want to see exactly what a cell senses? It's all in `truth.py`. Read it.

Good luck. Grow something that thinks. 🌱
