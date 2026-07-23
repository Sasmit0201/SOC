"""
============================================================================
  grow_nca.py  —  GROWING NEURAL CELLULAR AUTOMATA, from scratch (numpy)
============================================================================
A learning playground (not the competition). Here gradient descent REALLY works,
because the whole thing is differentiable — so you write the forward pass AND
backprop-through-time by hand, and watch an NCA grow from a single seed into a
target shape while the loss goes DOWN. This is Week 2, done from scratch.

    python grow_nca.py            # trains, saves grow_nca.png (target vs grown)

No PyTorch, no TensorFlow. Just numpy, the chain rule, and a tiny Adam optimiser
you can read. Once you *feel* how backprop through many steps works here, you
understand exactly what a framework does for you.
"""

import numpy as np

# --- the world ---
G = 24            # grid size
C = 8             # channels per cell (channel 0 = the visible image; rest hidden)
STEPS = 20        # how many CA steps we grow for
HID = 32          # hidden units in the update net
ITERS = 600       # training steps
LR = 3e-3

# fixed perception filters (identity + Sobel gradients), same idea as Truth Arena
IDENT = np.zeros((3, 3), np.float32); IDENT[1, 1] = 1.0
SOBX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
SOBY = SOBX.T.copy()


def conv(f, k):                      # 3x3 filter over the grid (wrap-around)
    out = np.zeros_like(f)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            if k[di, dj]:
                out += k[di, dj] * np.roll(np.roll(f, di - 1, 0), dj - 1, 1)
    return out


def rot180(k):
    return k[::-1, ::-1].copy()


def perceive(state):                 # (G,G,C) -> (G,G,3C): [state, d/dx, d/dy]
    return np.concatenate([conv(state, IDENT), conv(state, SOBX), conv(state, SOBY)], 2)


def perceive_adjoint(d_perc):        # the transpose of perceive, for backprop
    d0 = d_perc[:, :, :C]; d1 = d_perc[:, :, C:2 * C]; d2 = d_perc[:, :, 2 * C:]
    return conv(d0, rot180(IDENT)) + conv(d1, rot180(SOBX)) + conv(d2, rot180(SOBY))


def target_heart():
    y, x = np.mgrid[0:G, 0:G].astype(np.float32)
    x = (x - G / 2 + 0.5) / (G * 0.33); y = (G / 2 - 0.5 - y) / (G * 0.33)
    inside = (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 <= 0
    return inside.astype(np.float32)


# ---- the update net (its weights are what we train) ----
rng = np.random.default_rng(0)
W0 = (rng.standard_normal((3 * C, HID)) * np.sqrt(2 / (3 * C))).astype(np.float32)
b0 = np.zeros(HID, np.float32)
W1 = (rng.standard_normal((HID, C)) * 0.1).astype(np.float32)
b1 = np.zeros(C, np.float32)
# Adam state
mW0 = np.zeros_like(W0); vW0 = np.zeros_like(W0); mb0 = np.zeros_like(b0); vb0 = np.zeros_like(b0)
mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)


def seed():
    s = np.zeros((G, G, C), np.float32)
    s[G // 2, G // 2, :] = 1.0          # one living cell in the middle
    return s


def forward(target):
    """Grow for STEPS steps; remember what we need for backprop; return loss."""
    state = seed()
    tape = []
    for _ in range(STEPS):
        X = perceive(state).reshape(-1, 3 * C)     # (N, 3C)
        z1 = X @ W0 + b0
        a1 = np.maximum(z1, 0.0)
        dstate = (a1 @ W1 + b1).reshape(G, G, C)
        tape.append((X, z1, a1))
        state = state + dstate                     # residual update
    diff = state[:, :, 0] - target
    loss = float((diff ** 2).mean())
    return loss, state, tape, diff


def backward(state, tape, diff):
    """Backprop-through-time: from the loss, back through every step, to the
    weights. This is the whole point — the chain rule, by hand, over STEPS."""
    gW0 = np.zeros_like(W0); gb0 = np.zeros_like(b0)
    gW1 = np.zeros_like(W1); gb1 = np.zeros_like(b1)

    d_state = np.zeros((G, G, C), np.float32)
    d_state[:, :, 0] = 2.0 * diff / diff.size      # dLoss/d(final visible channel)

    for X, z1, a1 in reversed(tape):
        # state_next = state + dstate  ->  gradient wrt dstate is just d_state
        d_dstate = d_state.reshape(-1, C)
        gW1 += a1.T @ d_dstate; gb1 += d_dstate.sum(0)
        d_a1 = d_dstate @ W1.T
        d_z1 = d_a1 * (z1 > 0)                      # ReLU derivative
        gW0 += X.T @ d_z1; gb0 += d_z1.sum(0)
        d_X = (d_z1 @ W0.T).reshape(G, G, 3 * C)
        # gradient flows to state_t two ways: the residual skip, and through perceive
        d_state = d_state + perceive_adjoint(d_X)
    return gW0, gb0, gW1, gb1


def adam(p, g, m, v, t):
    m[:] = 0.9 * m + 0.1 * g
    v[:] = 0.999 * v + 0.001 * g * g
    mhat = m / (1 - 0.9 ** t); vhat = v / (1 - 0.999 ** t)
    p -= LR * mhat / (np.sqrt(vhat) + 1e-8)


def main():
    global W0, b0, W1, b1
    target = target_heart()
    for it in range(1, ITERS + 1):
        loss, state, tape, diff = forward(target)
        gW0, gb0, gW1, gb1 = backward(state, tape, diff)
        # normalise gradients (NCA training is sensitive) then Adam step
        for g in (gW0, gb0, gW1, gb1):
            g /= (np.linalg.norm(g) + 1e-8)
        adam(W0, gW0, mW0, vW0, it); adam(b0, gb0, mb0, vb0, it)
        adam(W1, gW1, mW1, vW1, it); adam(b1, gb1, mb1, vb1, it)
        if it % 50 == 0 or it == 1:
            print(f"iter {it:4d}   loss {loss:.4f}")

    # render: target vs what grew
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, final, _, _ = forward(target)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2)); fig.patch.set_facecolor("#0a0d11")
    for a, img, t in [(ax[0], target, "target"),
                      (ax[1], np.clip(final[:, :, 0], 0, 1), "grew from 1 seed")]:
        a.imshow(img, cmap="magma"); a.set_xticks([]); a.set_yticks([])
        a.set_title(t, color="#e9eef4")
    fig.tight_layout(); fig.savefig("grow_nca.png", dpi=120, facecolor="#0a0d11")
    plt.close(fig)

    # a gif of it growing from the single seed
    from matplotlib import animation
    state = seed(); frames = [np.clip(state[:, :, 0], 0, 1).copy()]
    for _ in range(STEPS):
        X = perceive(state).reshape(-1, 3 * C)
        dstate = (np.maximum(X @ W0 + b0, 0.0) @ W1 + b1).reshape(G, G, C)
        state = state + dstate
        frames.append(np.clip(state[:, :, 0], 0, 1).copy())
    figg, axg = plt.subplots(figsize=(3, 3)); figg.patch.set_facecolor("#0a0d11")
    axg.set_xticks([]); axg.set_yticks([])
    im = axg.imshow(frames[0], cmap="magma", vmin=0, vmax=1)
    animation.FuncAnimation(figg, lambda i: (im.set_data(frames[i]),), frames=len(frames),
                            interval=180).save("grow_nca.gif", writer=animation.PillowWriter(fps=6))
    plt.close(figg)
    print("saved grow_nca.png and grow_nca.gif")


if __name__ == "__main__":
    main()
