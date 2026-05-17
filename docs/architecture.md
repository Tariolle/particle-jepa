# Particle-JEPA Architecture Notes

## Goal

Particle-JEPA is a graph world model for particle dynamics. The context is the current particle graph `G_t`; the target is the next particle graph `G_t+1`.

The current graph contains positions and velocities, so unlike image-only Atari-style world models, it is already a Markov state for the toy simulator. We do not need frame stacking unless a dataset hides velocity or contains history-dependent material effects.

## Comparison Set

The project should compare:

1. **Particle-JEPA**: graph-native latent next-state prediction with SIGReg.
2. **GNS baseline**: official Graph Network Simulator-style dynamics prediction.

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
node latent prediction loss
+ spatial region latent prediction loss
+ graph latent prediction loss
+ SIGReg anti-collapse loss
```

The node loss keeps the objective particle-aware. The spatial region loss avoids the destructive "everything averages together" failure mode of pure mean pooling. The graph loss remains a coarse global alignment diagnostic. SIGReg prevents low-variance latent collapse without an EMA target encoder.

There is no decoded-state loss in the Particle-JEPA objective. Decoders or probes are trained only after the JEPA is frozen, as an evaluation tool.

## How We Know The Idea Is Working

The idea is not validated by low training loss alone. It becomes grounded when:

- frozen probes can decode useful future particle states from predicted latents,
- probe rollouts show plausible gravity, contact, and long-horizon behavior,
- retrieval top-k accuracy beats chance and random-latent baselines as a secondary diagnostic,
- latent standard deviation is not near zero,
- predicted-vs-target latent cosine is high but not due to collapse.

The threshold for comparing against GNS is:

```text
Frozen Particle-JEPA probe rollout is physically plausible on WaterRamps
```

GNS remains the reference baseline, not the project center.
