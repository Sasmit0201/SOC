"""Truth Arena — the world (your practice sandbox).

You can read this file to see EXACTLY what a cell senses and how liars, food, and
immunity work — you never need to edit it. The real competition is scored by the
host on hidden settings (see the note at the bottom).

A society of TRUE cells, driven by the student's Neural Cellular Automaton rule,
grows on FOOD while LIARS spread by conversion. A TRUE cell touching liars flips
UNLESS it has enough TRUE neighbours (immunity = cohesion). Over 70 steps the
biggest, most cohesive society that survives the liars wins.

Everything a student needs to TRAIN is a "rules of the game" default here; the
values that actually decide the competition (liar aggression, seeds) live in a
HIDDEN config that only the host has (see host score.py), never in this file.
"""

from dataclasses import dataclass, replace

import numpy as np

# ---- cell kinds ----
EMPTY, TRUE, LIAR = 0, 1, 2

# ---- the frozen contract (shared with students) --------------------------
# A cell perceives 7 fields, each through 3 fixed filters (identity + Sobel x/y):
#   [true, liar, food, energy, mem0, mem1, mem2]
N_FIELDS = 7
N_PERC = N_FIELDS * 3                 # 21 perception numbers per cell
N_MEM = 3                             # hidden memory channels the rule controls
# the submitted network: perception -> hidden -> [grow, dmem0, dmem1, dmem2]
RULE_SIZES = [N_PERC, 24, 1 + N_MEM]

_SOBX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
_SOBY = _SOBX.T.copy()
_ONES = np.ones((3, 3), np.float32)


@dataclass
class Config:
    grid: int = 48
    steps: int = 70
    # food
    n_food: int = 16           # food source patches (each a small cluster)
    food_regen: float = 0.08   # per-step regrowth at a source
    food_gain: float = 0.13    # energy a TRUE cell harvests from food under it
    food_cap: float = 1.0
    # energy economy: a cell earns a trickle, burns metabolism, thrives on food.
    # off food it slowly starves -> the society is CAPPED by the food it holds.
    passive_energy: float = 0.008
    metabolism: float = 0.020  # net -0.012/step off food -> must reach food
    strong_energy: float = 0.30  # below this a cell is "weak": loses immunity
    grow_thresh: float = 0.5   # grow-signal needed to reproduce
    grow_cost: float = 0.20
    child_energy: float = 0.28  # children start with enough to bridge a small gap
    # liars
    n_liar_seeds: int = 4      # starting liar clusters
    liar_wave_every: int = 15  # a fresh liar cluster arrives this often
    liar_strength: float = 0.62  # conversion pressure on exposed/weak cells
    immunity: int = 5          # TRUE neighbours (of 8) needed to be immune
    liar_decay: float = 0.05   # a liar with no TRUE neighbour may fade
    seed: int = 0


# ------------------------------------------------------------- grid helpers
def _conv3(f, k):
    out = np.zeros_like(f)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            if k[di, dj]:
                out += k[di, dj] * np.roll(np.roll(f, di - 1, 0), dj - 1, 1)
    return out


def _neighbours(mask):                       # count of 8 neighbours that are set
    return _conv3(mask.astype(np.float32), _ONES) - mask


def _dilate(mask):                           # True where a neighbour is set
    return _neighbours(mask) > 0


# --------------------------------------------------------------- the rule
class TruthRule:
    """The submission: the update net every TRUE cell runs."""

    def __init__(self, sizes=RULE_SIZES, seed=0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.params = []
        for i, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            W = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
            if i == len(self.sizes) - 2:
                W *= 0.0                       # start calm, not chaotic
            self.params.append([W, np.zeros(b, np.float32)])

    def forward(self, perc):
        a = perc.reshape(-1, self.sizes[0])
        for i, (W, b) in enumerate(self.params):
            a = a @ W + b
            if i < len(self.params) - 1:
                a = np.maximum(a, 0.0)
        return a.reshape(perc.shape[0], perc.shape[1], self.sizes[-1])

    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in self.params)

    def get_flat(self):
        return np.concatenate([x.ravel() for W, b in self.params for x in (W, b)])

    def set_flat(self, v):
        i = 0
        for W, b in self.params:
            for x in (W, b):
                x.flat[:] = v[i:i + x.size]; i += x.size
        return self

    def save(self, path, note=""):
        # `note` is your design note — a couple of sentences on why your society
        # survives. It travels inside the submission and is required to submit.
        np.savez(path, kind="truth-nca", sizes=np.array(self.sizes),
                 flat=self.get_flat(), note=np.array(str(note)))

    @classmethod
    def load(cls, path):
        p = path if str(path).endswith(".npz") else str(path) + ".npz"
        d = np.load(p, allow_pickle=False)
        r = cls(sizes=[int(x) for x in d["sizes"]])
        return r.set_flat(d["flat"])


def read_note(path):
    """The design note stored inside a submission (empty string if none)."""
    p = path if str(path).endswith(".npz") else str(path) + ".npz"
    d = np.load(p, allow_pickle=False)
    return str(d["note"]) if "note" in d.files else ""


# --------------------------------------------------------------- the world
class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        G = cfg.grid
        self.owner = np.zeros((G, G), np.int8)
        self.energy = np.zeros((G, G), np.float32)
        self.mem = np.zeros((G, G, N_MEM), np.float32)
        self.food = np.zeros((G, G), np.float32)
        self.src = np.zeros((G, G), bool)          # food regrows here
        self._setup()

    def _blob(self, r, c, rad, kind):
        G = self.cfg.grid
        yy, xx = np.ogrid[:G, :G]
        m = ((yy - r) ** 2 + (xx - c) ** 2) <= rad * rad
        self.owner[m] = kind
        if kind == TRUE:
            self.energy[m] = 0.6

    def _setup(self):
        G = self.cfg.grid
        # food sources scattered around, each a 5x5 patch (sustains a cluster)
        for _ in range(self.cfg.n_food):
            r, c = self.rng.integers(4, G - 4, size=2)
            self.src[r - 2:r + 3, c - 2:c + 3] = True
            self.food[r - 2:r + 3, c - 2:c + 3] = self.cfg.food_cap
        # the TRUE society starts as a small blob near the centre, on some food
        cx = G // 2
        self._blob(cx, cx, 2, TRUE)
        self.src[cx, cx] = True; self.food[cx, cx] = self.cfg.food_cap
        # liar seeds around the edges
        for _ in range(self.cfg.n_liar_seeds):
            edge = self.rng.integers(0, 4)
            r = self.rng.integers(3, G - 3)
            pos = {0: (3, r), 1: (G - 4, r), 2: (r, 3), 3: (r, G - 4)}[edge]
            self._blob(pos[0], pos[1], 1, LIAR)

    # -- perception fields the rule reads -------------------------------
    def _fields(self):
        true = (self.owner == TRUE).astype(np.float32)
        liar = (self.owner == LIAR).astype(np.float32)
        return np.stack([true, liar, self.food, self.energy,
                         self.mem[:, :, 0], self.mem[:, :, 1], self.mem[:, :, 2]], axis=2)

    def _perceive(self, fields):
        return np.concatenate([fields, _conv3(fields, _SOBX),
                               _conv3(fields, _SOBY)], axis=2)

    # -- one step -------------------------------------------------------
    def step(self, rule, t):
        cfg = self.cfg
        true = self.owner == TRUE

        # 1) food regrows; a TRUE cell earns a trickle, burns metabolism, feasts on
        #    food. Off food it slowly starves; at zero energy it dies.
        self.food[self.src] = np.minimum(cfg.food_cap, self.food[self.src] + cfg.food_regen)
        self.energy[true] += cfg.passive_energy - cfg.metabolism
        harvest = true & (self.food > 0)
        self.energy[harvest] += cfg.food_gain
        self.food[harvest] = np.maximum(0.0, self.food[harvest] - cfg.food_gain)
        np.clip(self.energy, 0.0, 1.0, out=self.energy)
        starved = true & (self.energy <= 0)
        self.owner[starved] = EMPTY
        self.mem[starved] = 0.0
        true = self.owner == TRUE

        # 2) the rule updates memory + emits a grow signal
        out = rule.forward(self._perceive(self._fields()))
        grow_sig = 1.0 / (1.0 + np.exp(-out[:, :, 0]))         # sigmoid
        upd = (self.rng.random((cfg.grid, cfg.grid, 1)) < 0.5)  # async
        self.mem += upd * out[:, :, 1:] * true[:, :, None]
        np.clip(self.mem, -3.0, 3.0, out=self.mem)

        # 3) growth: an empty cell beside an eligible TRUE cell becomes TRUE
        eligible = true & (self.energy > cfg.grow_cost) & (grow_sig > cfg.grow_thresh)
        born = (self.owner == EMPTY) & _dilate(eligible)
        self.owner[born] = TRUE
        self.energy[born] = cfg.child_energy
        self.energy[eligible & _dilate(born)] -= cfg.grow_cost
        np.clip(self.energy, 0.0, 1.0, out=self.energy)

        # 4) liars convert exposed cells. Immunity needs BOTH density AND energy:
        #    a starving cell (weak, uninformed) resists far worse even when dense.
        true = self.owner == TRUE
        liar = self.owner == LIAR
        exposed = true & _dilate(liar)
        support = _neighbours(true)                            # TRUE neighbours (0..8)
        strong = self.energy > cfg.strong_energy
        eff = support * np.where(strong, 1.0, 0.4)             # weak cells count for less
        prob = np.clip((cfg.immunity - eff) / cfg.immunity, 0.0, 1.0) * cfg.liar_strength
        flip = exposed & (self.rng.random((cfg.grid, cfg.grid)) < prob)
        self.owner[flip] = LIAR
        self.energy[flip] = 0.0
        self.mem[flip] = 0.0

        # 5) liars with nothing to feed on slowly fade
        lonely = (self.owner == LIAR) & ~_dilate(self.owner == TRUE)
        fade = lonely & (self.rng.random((cfg.grid, cfg.grid)) < cfg.liar_decay)
        self.owner[fade] = EMPTY

        # 6) a fresh liar wave every so often: strike the FRONTIER from outside,
        #    so a well-defended edge can hold it (hidden aggression).
        if cfg.liar_wave_every and t > 0 and t % cfg.liar_wave_every == 0:
            frontier = np.argwhere((self.owner == EMPTY) & _dilate(self.owner == TRUE))
            if len(frontier):
                r, c = frontier[self.rng.integers(len(frontier))]
                self._blob(int(r), int(c), 1, LIAR)

    # -- scoring --------------------------------------------------------
    def score(self):
        true = self.owner == TRUE
        n = int(true.sum())
        size = n / (self.cfg.grid ** 2)
        cohesion = float((_neighbours(true)[true] / 8.0).mean()) if n else 0.0
        return {"score": size * cohesion, "size": n, "cohesion": cohesion,
                "coverage": size}


def run(rule, cfg=None, seed=0, record=False):
    cfg = replace(cfg or Config(), seed=seed)
    w = World(cfg)
    frames = [] if record else None
    if record:
        frames.append((w.owner.copy(), w.food.copy()))
    for t in range(cfg.steps):
        w.step(rule, t)
        if record:
            frames.append((w.owner.copy(), w.food.copy()))
    out = w.score()
    out["frames"] = frames
    return out


def evaluate(rule, seeds=(0, 1, 2), cfg_fn=None):
    """Mean score over runs. Students use the practice Config; the host passes its
    own private cfg_fn (unseen seeds & difficulty) for the real evaluation."""
    cfg_fn = cfg_fn or (lambda s: Config(seed=s))
    rs = [run(rule, cfg=cfg_fn(s), seed=s) for s in seeds]
    return {"score": float(np.mean([r["score"] for r in rs])),
            "size": float(np.mean([r["size"] for r in rs])),
            "cohesion": float(np.mean([r["cohesion"] for r in rs]))}
