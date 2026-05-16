# Particle-JEPA Architecture Notes

## Goal

Particle-JEPA is a graph world model for particle dynamics. The context is the current particle graph `G_t`; the target is the next particle graph `G_t+1`.

The current graph contains positions and velocities, so unlike image-only Atari-style world models, it is already a Markov state for the toy simulator. We do not need frame stacking unless a dataset hides velocity or contains history-dependent material effects.

## Comparison Set

The project should compare:

1. **Particle-JEPA**: graph-native latent next-state prediction with SIGReg.
2. **GNS baseline**: official Graph Network Simulator-style dynamics prediction.
3. **Hybrid GNS + JEPA**: supervised dynamics prediction plus the Particle-JEPA auxiliary objective.

## Encoder Options

### GNS / MeshGraphNet-Style MPNN

This is the safest default and the closest architecture to the learned simulation literature.

```text
node features -> node MLP
edge features -> edge MLP
K rounds of message passing
node latents + pooled graph latent
```

Strengths:

- Strong local interaction bias.
- Natural use of edge geometry.
- Directly comparable to GNS and MeshGraphNets.
- Efficient on dynamic radius graphs.

Weaknesses:

- Long-range effects require many message-passing steps.
- Not explicitly equivariant beyond relative geometric features.

### Equivariant GNN

EGNN-style models explicitly respect Euclidean symmetries.

Strengths:

- Better inductive bias for physics.
- Potentially better sample efficiency and generalization.

Weaknesses:

- More architectural complexity.
- Box boundaries, gravity, and material flags can break full rotational symmetry.

### Graph Transformer / GraphGPS

GraphGPS combines local message passing with global attention and positional or structural encodings.

Strengths:

- Captures global interactions.
- Can represent long-range dependencies more directly.

Weaknesses:

- More expensive.
- Less physically biased unless edge geometry is carefully injected.
- Likely overkill before the local graph world model is validated.

## Chosen Encoder

Use a GNS/MeshGraphNet-style MPNN encoder first.

```text
G_t -> MPNN encoder -> latent graph H_t
G_t+1 -> same MPNN encoder -> latent graph H_t+1
```

This is grounded in learned physics simulation and gives a fair bridge to the GNS baseline.

## Predictor Options

### MLP Predictor

The first MVP used:

```text
node latent + horizon embedding -> predicted node latent
graph latent + horizon embedding -> predicted graph latent
```

This was useful as a smoke test, but it is too weak as the main architecture because each node is updated independently after encoding.

### Latent Message-Passing Predictor

The next architecture uses graph-native latent dynamics:

```text
latent node states H_t
current graph edges E_t
edge latents from edge features
message passing in latent space
predicted latent graph H_hat_t+1
```

This is the preferred predictor because it models the next latent state through relational particle interactions.

### Transformer Predictor

A transformer predictor is closer to LeWM's image-world-model architecture.

```text
particle latent tokens -> transformer -> predicted next latent tokens
```

This is a later option if latent message passing is insufficient.

## Chosen Predictor

Use a latent graph message-passing predictor:

```text
G_t
  -> MPNN encoder
  -> latent graph H_t
  -> latent MPNN predictor
  -> predicted latent graph H_hat_t+1

G_t+1
  -> same MPNN encoder
  -> target latent graph H_t+1
```

## Loss

Particle-JEPA uses:

```text
graph latent prediction loss
+ node latent prediction loss
+ SIGReg anti-collapse loss
```

The graph loss supports retrieval and trajectory metrics. The node loss keeps the objective graph-native and particle-aware. SIGReg prevents low-variance latent collapse.

## How We Know The Idea Is Working

The idea is not validated by low training loss alone. It becomes grounded when:

- retrieval top-k accuracy beats chance and random-latent baselines,
- latent standard deviation is not near zero,
- predicted-vs-target latent cosine is high but not due to collapse,
- hybrid GNS + JEPA improves rollout metrics over GNS alone,
- video rollouts show lower long-horizon drift or better stability.

The threshold for moving beyond toy image panels into video comparisons is:

```text
Particle-JEPA retrieval lift > 1 over chance
and
Hybrid rollout error <= GNS rollout error on the same split
```

Once both are true on toy data, video comparison becomes meaningful rather than decorative.
