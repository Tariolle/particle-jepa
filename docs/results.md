# Experiment Notes

## 2026-05-16: Latent Message-Passing Particle-JEPA

Configuration:

```text
data.num_train_trajectories=16
data.num_val_trajectories=4
data.trajectory_length=12
data.num_particles=16
data.horizon=1
training.epochs=8
training.compile=true
training.compile_mode=reduce-overhead
model.hidden_dim=64
model.latent_dim=64
model.message_passing_steps=2
model.latent_predictor_steps=2
loss.sigreg_weight=0.05
```

Retrieval evaluation:

```json
{
  "top1_accuracy": 0.0454545468,
  "top5_accuracy": 0.3106060624,
  "chance_top1_accuracy": 0.0075757576,
  "chance_top5_accuracy": 0.0378787879,
  "random_top1_accuracy": 0.0069839018,
  "random_top5_accuracy": 0.0334990546,
  "num_samples": 132,
  "prediction_target_cosine": 0.9718736410,
  "prediction_latent_std": 0.0978704616,
  "target_latent_std": 0.0995207727,
  "top1_lift_vs_chance": 6.0,
  "top5_lift_vs_chance": 8.2
}
```

Interpretation:

- The latent graph predictor gives a clear retrieval signal above chance.
- Latent standard deviations are not collapsed.
- This supports the Particle-JEPA representation-learning idea on toy data.
- It does not yet prove the world-modeling claim. The next required result is whether **Hybrid GNS + JEPA** improves rollout metrics over **GNS** on the same split.
