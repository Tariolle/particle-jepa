# Architecture Notes

This document records the architecture used in the closed Particle-JEPA
prototype and why it was not sufficient.

## Implemented Model

Particle-JEPA used a shared particle-graph encoder for context and target
graphs:

```text
G_t   -> encoder -> context node/region/graph latents
G_t+1 -> encoder -> target node/region/graph latents
```

A latent predictor consumed the context graph latents and predicted future
latents:

```text
context node latents + graph structure + horizon
  -> latent predictor
  -> predicted future node/region/graph latents
```

The encoder was GNS/MeshGraphNet-style message passing. The predictor was tested
with graph-native latent prediction, including transformer-style predictor
configs for WaterRamps experiments.

## Loss

The intended objective was JEPA-style latent alignment plus SIGReg:

```text
node latent prediction
+ region latent prediction
+ graph latent prediction
+ SIGReg anti-collapse
```

Late diagnostic runs also tested latent delta consistency. This did not change
the outcome: the frozen latents still were not useful enough for physical
rollout.

## Probe Evaluation

The decoder/probe was deliberately lightweight. It was not meant to be the
simulator; it measured whether a frozen latent contained easily recoverable
dynamics information.

The important probes were:

```text
raw_features     -> acceleration
node_context     -> acceleration
node_prediction  -> acceleration
```

The failing `node_context` result is the most important one: even before the
predictor, the JEPA encoder latent did not behave like a dynamics-sufficient
state.

## Why It Failed

Latent agreement does not identify simulator state. The objective can produce
non-collapsed, predictable latents while losing details that matter for rollout:

- exact velocity information,
- gravity direction in a decodable coordinate frame,
- contact geometry,
- boundary relation,
- particle type effects,
- neighbor interactions needed for pressure/contact.

Video JEPA can tolerate semantic abstraction. Particle rollout cannot: small
acceleration errors compound immediately.

## Final Architectural Takeaway

The architecture was not obviously broken as code, but the objective was
underconstrained for the task. Continuing this line would require a different
SSL formulation, not just a larger predictor or decoder.
