# Final Results

Particle-JEPA was tested as a self-supervised graph world model for particle
simulation. The prototype is closed with a negative result.

## Main Conclusion

The JEPA/SIGReg objective did not learn particle latents that were sufficient for
rollout. A supervised GNS baseline worked decently on the same WaterRamps data,
so the failure is specific to the learned JEPA representation and predictor, not
to the basic data or rollout pipeline.

## Decisive Diagnostic

After training Particle-JEPA, the model was frozen and lightweight acceleration
probes were trained from three sources:

```text
raw_features:
  learned gravity, but not ramp contact
  rollout position error: 0.012546

node_context:
  gravity direction was physically wrong
  rollout position error: 0.217165

node_prediction:
  particle block lost coherence before contact
  rollout position error: 0.038809
```

Interpretation:

- `raw_features` shows the simple MLP probe can learn easy local effects such as
  gravity, but lacks relational capacity for contact.
- `node_context` shows the JEPA encoder latent is not a clean dynamics state.
- `node_prediction` shows the predictor makes the latent state even less
  physically coherent.

## Baseline

The corrected WaterRamps GNS baseline used official-style inputs:

- six-frame position history,
- normalized velocity history,
- clipped boundary-distance features,
- one-hot particle types,
- normalized relative displacement edges,
- dynamic-particle noise,
- kinematic-particle handling.

Observed reference run:

```text
train loss: 0.5530 -> 0.1161
contact rollout error: 0.0410
```

This baseline was not perfect, but it learned plausible contact behavior. That
is enough to rule out the broadest pipeline failures.

## Research Decision

Do not continue patching this exact Particle-JEPA objective with more auxiliary
losses. Adding acceleration or rollout supervision would mostly turn the method
into a supervised simulator with extra representation machinery.

Future work would need a different formulation, for example:

- JEPA only as pretraining for a supervised simulator,
- transition-sufficient SSL targets designed around relative motion/contact,
- stronger physics-biased or equivariant graph architectures,
- or simply supervised GNS-style training when rollout accuracy is the goal.
