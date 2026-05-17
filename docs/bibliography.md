# Particle-JEPA Bibliography

## Core JEPA

- Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**. I-JEPA predicts latent representations of missing image regions rather than reconstructing pixels.
  <https://arxiv.org/abs/2301.08243>
- Bardes et al., **Revisiting Feature Prediction for Learning Visual Representations from Video**. V-JEPA learns by predicting video features in latent space, not pixels.
  <https://arxiv.org/abs/2404.08471>

## Graph JEPA

- **Graph-JEPA** adapts JEPA to graphs through masked subgraph representation prediction. It is useful for graph-native target construction, but it is not a physical world model by itself.
  <https://arxiv.org/abs/2309.16014>

## Latent World Models

- **LeJEPA** motivates the clean loss we want: latent prediction plus SIGReg anti-collapse, without relying on EMA targets as the main stabilizer.
  <https://arxiv.org/abs/2511.08544>
- **LeWorldModel / LeWM** frames JEPA as a latent world model. Decoders are used after training as probes for human inspection and downstream evaluation, not as part of the JEPA loss.
  <https://arxiv.org/abs/2603.19312>
  <https://le-wm.github.io/>

## Learned Particle Simulation Baseline

- Sanchez-Gonzalez et al., **Learning to Simulate Complex Physics with Graph Networks**. This remains the supervised GNS reference baseline for WaterRamps and related datasets.
  <https://arxiv.org/abs/2002.09405>
  <https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate>

## Project Takeaway

Particle-JEPA should not train with decoded-state or reconstruction losses. The JEPA objective is:

```text
latent prediction loss + SIGReg anti-collapse
```

A decoder or dynamics probe is trained only after freezing the JEPA, so it can measure whether the latent dynamics contain enough information to recover useful physical futures.
