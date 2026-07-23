"""Shared rendering: turn a recorded run into frames / gifs. 🟢 truth 🔴 liars 🟡 food."""

import numpy as np

import truth as T


def rgb(owner, food):
    G = owner.shape[0]
    img = np.full((G, G, 3), 0.06, np.float32)
    fm = (owner == T.EMPTY) & (food > 0.05)
    img[fm] = np.stack([0.55 * food, 0.45 * food, 0.10 * food], -1)[fm]
    img[owner == T.TRUE] = (0.15, 0.90, 0.30)
    img[owner == T.LIAR] = (0.90, 0.20, 0.20)
    return img


def save_gif(frames, path, fps=12, title="Truth Arena"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks([]); ax.set_yticks([]); fig.patch.set_facecolor("#0a0d11")
    o0, f0 = frames[0]
    im = ax.imshow(rgb(o0, f0), interpolation="nearest")
    ttl = ax.set_title(title, color="white")

    def upd(i):
        o, f = frames[i]
        im.set_data(rgb(o, f))
        nt = int((o == T.TRUE).sum()); nl = int((o == T.LIAR).sum())
        ttl.set_text(f"{title}   t={i}   truth={nt}  liars={nl}")
        return im, ttl
    animation.FuncAnimation(fig, upd, frames=len(frames), interval=1000 / fps,
                            blit=False).save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path
