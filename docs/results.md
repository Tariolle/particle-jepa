# Experiment Notes

## Current Focus

Particle-JEPA is now the primary research object. The project goal is to test
whether a JEPA-style graph latent dynamics model can learn useful particle
world representations.

The GNS model remains as the supervised learned-simulation baseline. It is not
the project center.

## What Counts As Evidence

Retrieval alone is not enough. A nearest-neighbor panel can look bad even when
scalar retrieval metrics beat chance, and it can look plausible for shallow
reasons.

The stronger evaluation path is:

1. Train Particle-JEPA with latent prediction plus SIGReg only.
2. Freeze the JEPA encoder and predictor.
3. Train a lightweight probe from predicted future node latents to particle
   acceleration or next state.
4. Evaluate one-step probe error, rollout videos, and contact behavior.

No decoded-state loss belongs in JEPA training. The probe is only an evaluation
instrument.

## Baseline Reference

The corrected WaterRamps GNS baseline uses official-style inputs:

- six-frame position history,
- normalized velocity history,
- clipped boundary-distance features,
- one-hot particle types,
- normalized relative displacement edges,
- dynamic-particle noise,
- kinematic-particle handling.

Recent GNS WaterRamps contact rollout:

```text
train loss: 0.5530 -> 0.1161
contact rollout error: 0.0410
```

This is the reference bar for later probe-based Particle-JEPA evaluation.
