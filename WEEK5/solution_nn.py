"""
============================================================================
  solution_nn.py  —  TRAIN A NEURAL NET FROM SCRATCH (forward + backprop)
============================================================================
The ambitious path. No PyTorch, no TensorFlow — you write the network AND its
gradients yourself, in pure numpy, then train it with real gradient steps.

Because the score comes from *running the society* (not from labels), we can't
backprop through the world. So we use REINFORCE — the same policy-gradient idea
behind a Week-4 DQN: the net makes slightly-random decisions, we run the society,
and we push the weights toward the decisions that led to a better-than-usual run.
The gradient we push needs only backprop through the NET — which you own below.

Run:  python solution_nn.py   ->  my_rule.npz   ->  python check.py my_rule.npz

Note: RL is high-variance — this is the "understand it deeply" path and may not
beat solution.py (evolution). Both produce a valid my_rule.npz; pick your fun.
----------------------------------------------------------------------------
"""

import numpy as np

from truth import RULE_SIZES, TruthRule, run   # the world + the weight format

SIGMA    = 0.30    # how much randomness to add to actions (exploration)
EPISODES = 200     # how many societies to train on
LR       = 0.20    # learning rate

# >>> REQUIRED before submitting: 1-2 sentences on why your society survives. <<<
NOTE = "TODO — write why your society survives the liars (1-2 sentences)."


# ===========================================================================
#  THE NEURAL NET — forward pass and backprop, by hand.  (21 -> 24 -> 4)
# ===========================================================================
class Net:
    def __init__(self, seed=0):
        nin, nhid, nout = RULE_SIZES            # 21, 24, 4
        rng = np.random.default_rng(seed)
        self.W0 = (rng.standard_normal((nin, nhid)) * np.sqrt(2 / nin)).astype(np.float32)
        self.b0 = np.zeros(nhid, np.float32)
        self.W1 = np.zeros((nhid, nout), np.float32)   # start the output at 0 (calm)
        self.b1 = np.zeros(nout, np.float32)

    def forward(self, X):
        """X: (N, 21) -> out: (N, 4).  Also returns a 'cache' for backprop."""
        z1 = X @ self.W0 + self.b0
        a1 = np.maximum(z1, 0.0)                # ReLU
        out = a1 @ self.W1 + self.b1            # linear output
        return out, (X, z1, a1)

    def backward(self, cache, d_out):
        """The chain rule, by hand. d_out = d(objective)/d(out); returns the
        gradient for every weight."""
        X, z1, a1 = cache
        dW1 = a1.T @ d_out
        db1 = d_out.sum(axis=0)
        d_a1 = d_out @ self.W1.T
        d_z1 = d_a1 * (z1 > 0)                  # gradient flows only where ReLU was on
        dW0 = X.T @ d_z1
        db0 = d_z1.sum(axis=0)
        return dW0, db0, dW1, db1

    def add(self, grads, lr):                  # gradient ASCENT (maximize reward)
        dW0, db0, dW1, db1 = grads
        self.W0 += lr * dW0; self.b0 += lr * db0
        self.W1 += lr * dW1; self.b1 += lr * db1

    def flat(self):                            # weights in the arena's format
        return np.concatenate([self.W0.ravel(), self.b0, self.W1.ravel(), self.b1])


# ===========================================================================
#  A stochastic policy that the society engine can run. Each step it samples a
#  slightly-random action from the net and tapes what it did, so we can reward
#  the good runs afterwards.
# ===========================================================================
class Policy:
    def __init__(self, net, rng):
        self.net, self.rng, self.tape = net, rng, []

    def forward(self, perc):                   # the engine calls this every step
        H, W, C = perc.shape
        X = perc.reshape(-1, C)
        mean, cache = self.net.forward(X)
        action = mean + self.rng.standard_normal(mean.shape).astype(np.float32) * SIGMA
        true_cell = (perc[:, :, 0] > 0.5).reshape(-1)   # only TRUE cells actually act
        self.tape.append((cache, action, mean, true_cell))
        return action.reshape(H, W, -1)


# ===========================================================================
#  Training loop: REINFORCE.
# ===========================================================================
def train():
    net = Net()
    rng = np.random.default_rng(0)
    avg_R, var_R = 0.0, 1.0

    for ep in range(EPISODES):
        policy = Policy(net, rng)
        R = run(policy, seed=ep % 5)["score"]          # play one society

        # "advantage" = was this run better or worse than our recent average?
        avg_R = 0.95 * avg_R + 0.05 * R
        var_R = 0.95 * var_R + 0.05 * (R - avg_R) ** 2
        advantage = (R - avg_R) / (np.sqrt(var_R) + 1e-6)

        # sum the policy gradient over every decision in the run
        gW0 = np.zeros_like(net.W0); gb0 = np.zeros_like(net.b0)
        gW1 = np.zeros_like(net.W1); gb1 = np.zeros_like(net.b1)
        n_decisions = 0
        for cache, action, mean, true_cell in policy.tape:
            # d(log P(action)) / d(out) for a Gaussian policy = (action - mean)/sigma^2
            d_out = ((action - mean) / (SIGMA ** 2)) * true_cell[:, None]
            dW0, db0, dW1, db1 = net.backward(cache, d_out)
            gW0 += dW0; gb0 += db0; gW1 += dW1; gb1 += db1
            n_decisions += int(true_cell.sum())

        # push the weights toward the good actions (scaled by how good the run was)
        scale = LR * advantage / max(n_decisions, 1)
        net.add((gW0, gb0, gW1, gb1), scale)

        if ep % 20 == 0:
            print(f"episode {ep:3d}   reward {R:.3f}   avg {avg_R:.3f}")
    return net


if __name__ == "__main__":
    print("training a neural net from scratch (forward + backprop, numpy only)…")
    net = train()
    TruthRule().set_flat(net.flat()).save("my_rule", note=NOTE)
    print("\nsaved my_rule.npz   ->   python check.py my_rule.npz")
