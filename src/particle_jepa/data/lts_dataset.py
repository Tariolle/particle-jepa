from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from torch import Tensor
from torch.utils.data import Dataset

from particle_jepa.data.graph_builder import ParticleGraphBuilder

LTS_DATASETS = (
    "WaterDrop",
    "Water",
    "Sand",
    "Goop",
    "MultiMaterial",
    "RandomFloor",
    "WaterRamps",
    "SandRamps",
    "FluidShake",
    "FluidShakeBox",
    "Continuous",
    "WaterDrop-XL",
    "Water-3D",
    "Sand-3D",
    "Goop-3D",
    "WaterDropSample",
)

BASE_URL = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets"


@dataclass(frozen=True)
class LearningToSimulateConfig:
    """Configuration for DeepMind Learning-to-Simulate TFRecord trajectories."""

    root: str | Path = "data/raw/WaterDropSample"
    split: str = "train"
    future_offset: int = 1
    radius: float | None = None
    max_neighbors: int | None = None
    max_trajectories: int | None = None
    sample_stride: int = 1
    max_samples_per_trajectory: int | None = 128
    normalize_acceleration: bool = True
    kinematic_particle_id: int = 3


class LearningToSimulateDataset(Dataset):
    """Graph-pair dataset backed by DeepMind Learning-to-Simulate TFRecords.

    The raw SequenceExample records are decoded into trajectories with positions,
    finite-difference velocities, particle types, and optional per-step context.
    Each item returns `(G_t, G_t+k)` as PyG `Data` objects, matching the toy
    dataset interface used by the training code.
    """

    def __init__(self, config: LearningToSimulateConfig | str | Path) -> None:
        if isinstance(config, str | Path):
            config = LearningToSimulateConfig(root=config)
        self.config = config
        self.root = Path(config.root)
        self.metadata = load_metadata(self.root)
        self.dt = float(self.metadata.get("dt", 1.0))
        self.radius = float(config.radius or self.metadata["default_connectivity_radius"])
        self.graph_builder = ParticleGraphBuilder(
            radius=self.radius, max_neighbors=config.max_neighbors
        )
        self.trajectories = list(
            load_lts_trajectories(
                self.root,
                split=config.split,
                metadata=self.metadata,
                max_trajectories=config.max_trajectories,
            )
        )
        if not self.trajectories:
            msg = f"No LTS trajectories found in {self.root / f'{config.split}.tfrecord'}."
            raise FileNotFoundError(msg)
        self.indices = self._build_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        traj_idx, time_idx = self.indices[index]
        trajectory = self.trajectories[traj_idx]
        future_idx = time_idx + self.config.future_offset

        particle_type = trajectory["particle_types"]
        context = self.graph_builder.build(
            trajectory["positions"][time_idx],
            trajectory["velocities"][time_idx],
            particle_type=particle_type,
            boundary=self._boundary_flags(trajectory["positions"][time_idx]),
        )
        future = self.graph_builder.build(
            trajectory["positions"][future_idx],
            trajectory["velocities"][future_idx],
            particle_type=particle_type,
            boundary=self._boundary_flags(trajectory["positions"][future_idx]),
        )

        next_idx = min(time_idx + 1, trajectory["positions"].size(0) - 1)
        next_position = trajectory["positions"][next_idx]
        next_velocity = trajectory["velocities"][next_idx]
        raw_acceleration = next_velocity - context.velocity
        acceleration = self._normalize_acceleration(raw_acceleration)
        dynamic_mask = particle_type != self.config.kinematic_particle_id

        context.y_pos = next_position
        context.y_velocity = next_velocity
        context.y_acceleration_raw = raw_acceleration
        context.y_acceleration = acceleration
        context.dynamic_mask = dynamic_mask.float()
        context.kinematic_mask = (~dynamic_mask).float()
        context.horizon = torch.tensor([self.config.future_offset], dtype=torch.long)
        future.horizon = torch.tensor([self.config.future_offset], dtype=torch.long)
        if "step_context" in trajectory:
            context.step_context = trajectory["step_context"][time_idx]
            future.step_context = trajectory["step_context"][future_idx]
        return context, future

    def _build_indices(self) -> list[tuple[int, int]]:
        indices: list[tuple[int, int]] = []
        stride = max(int(self.config.sample_stride), 1)
        for traj_idx, trajectory in enumerate(self.trajectories):
            max_t = trajectory["positions"].size(0) - self.config.future_offset
            time_indices = list(range(0, max_t, stride))
            if self.config.max_samples_per_trajectory is not None:
                time_indices = time_indices[: self.config.max_samples_per_trajectory]
            indices.extend((traj_idx, time_idx) for time_idx in time_indices)
        return indices

    def _boundary_flags(self, positions: Tensor) -> Tensor:
        bounds = self.metadata.get("bounds")
        if bounds is None:
            return torch.zeros(positions.size(0), dtype=torch.float32)
        bounds_tensor = torch.tensor(bounds, dtype=positions.dtype, device=positions.device)
        lower = bounds_tensor[:, 0]
        upper = bounds_tensor[:, 1]
        near_lower = positions <= lower + self.radius
        near_upper = positions >= upper - self.radius
        return (near_lower | near_upper).any(dim=-1).float()

    def _normalize_acceleration(self, acceleration: Tensor) -> Tensor:
        if not self.config.normalize_acceleration:
            return acceleration
        mean = torch.tensor(self.metadata["acc_mean"], dtype=acceleration.dtype)
        std = torch.tensor(self.metadata["acc_std"], dtype=acceleration.dtype).clamp_min(1e-8)
        return (acceleration - mean) / std


def load_metadata(root: str | Path) -> dict:
    """Load an LTS `metadata.json` file."""
    metadata_path = Path(root) / "metadata.json"
    if not metadata_path.exists():
        msg = f"Missing LTS metadata file: {metadata_path}"
        raise FileNotFoundError(msg)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_lts_trajectories(
    root: str | Path,
    split: str,
    metadata: dict | None = None,
    max_trajectories: int | None = None,
) -> Iterator[dict[str, Tensor | dict]]:
    """Yield decoded trajectories from an LTS TFRecord split."""
    root = Path(root)
    metadata = metadata or load_metadata(root)
    path = root / f"{split}.tfrecord"
    for record_idx, payload in enumerate(iter_tfrecord_records(path)):
        if max_trajectories is not None and record_idx >= max_trajectories:
            break
        yield decode_lts_sequence_example(payload, metadata)


def iter_tfrecord_records(path: str | Path) -> Iterator[bytes]:
    """Yield raw payloads from an uncompressed TFRecord file."""
    path = Path(path)
    if not path.exists():
        msg = f"Missing TFRecord file: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                msg = f"Corrupt TFRecord length header in {path}"
                raise ValueError(msg)
            (length,) = struct.unpack("<Q", length_bytes)
            handle.read(4)
            payload = handle.read(length)
            if len(payload) != length:
                msg = f"Corrupt TFRecord payload in {path}"
                raise ValueError(msg)
            handle.read(4)
            yield payload


def decode_lts_sequence_example(payload: bytes, metadata: dict) -> dict[str, Tensor | dict]:
    """Decode one LTS SequenceExample payload into trajectory tensors."""
    example = _sequence_example_class()()
    example.ParseFromString(payload)

    feature_lists = example.feature_lists.feature_list
    positions = _decode_float_feature_list(feature_lists["position"].feature)
    dim = int(metadata["dim"])
    positions = positions.reshape(int(metadata["sequence_length"]) + 1, -1, dim)

    particle_types = _decode_int64_bytes_feature(
        example.context.feature["particle_type"].bytes_list.value
    )
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    velocities[0] = velocities[1]

    trajectory: dict[str, Tensor | dict] = {
        "positions": torch.from_numpy(positions.astype(np.float32)),
        "velocities": torch.from_numpy(velocities.astype(np.float32)),
        "particle_types": torch.from_numpy(particle_types.astype(np.int64)),
        "metadata": metadata,
    }
    if "context_mean" in metadata and "step_context" in feature_lists:
        context = _decode_float_feature_list(feature_lists["step_context"].feature)
        context_dim = len(metadata.get("context_mean", []))
        if context_dim > 0:
            trajectory["step_context"] = torch.from_numpy(
                context.reshape(positions.shape[0], context_dim).astype(np.float32)
            )
    return trajectory


def _decode_float_feature_list(features) -> np.ndarray:
    arrays = []
    for feature in features:
        arrays.append(np.frombuffer(feature.bytes_list.value[0], dtype=np.float32))
    return np.asarray(arrays)


def _decode_int64_bytes_feature(values) -> np.ndarray:
    arrays = [np.frombuffer(value, dtype=np.int64) for value in values]
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


@lru_cache(maxsize=1)
def _sequence_example_class():
    pool = _proto_pool()
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.SequenceExample"))


@lru_cache(maxsize=1)
def _proto_pool() -> descriptor_pool.DescriptorPool:
    pool = descriptor_pool.DescriptorPool()
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "particle_jepa_tensorflow_example.proto"
    file_descriptor.package = "tensorflow"
    file_descriptor.syntax = "proto3"

    _add_list_message(file_descriptor, "BytesList", "bytes", 12)
    _add_list_message(file_descriptor, "FloatList", "float", 2)
    _add_list_message(file_descriptor, "Int64List", "int64", 3)
    _add_feature_messages(file_descriptor)
    pool.Add(file_descriptor)
    return pool


def _add_list_message(file_descriptor, name: str, type_name: str, field_type: int) -> None:
    message = file_descriptor.message_type.add()
    message.name = name
    field = message.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = field_type


def _add_feature_messages(file_descriptor) -> None:
    feature = file_descriptor.message_type.add()
    feature.name = "Feature"
    for number, (name, type_name) in enumerate(
        [
            ("bytes_list", ".tensorflow.BytesList"),
            ("float_list", ".tensorflow.FloatList"),
            ("int64_list", ".tensorflow.Int64List"),
        ],
        start=1,
    ):
        field = feature.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = type_name

    features = file_descriptor.message_type.add()
    features.name = "Features"
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    _add_map_key_value(entry, ".tensorflow.Feature")
    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features.FeatureEntry"

    feature_list = file_descriptor.message_type.add()
    feature_list.name = "FeatureList"
    field = feature_list.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Feature"

    feature_lists = file_descriptor.message_type.add()
    feature_lists.name = "FeatureLists"
    entry = feature_lists.nested_type.add()
    entry.name = "FeatureListEntry"
    entry.options.map_entry = True
    _add_map_key_value(entry, ".tensorflow.FeatureList")
    field = feature_lists.field.add()
    field.name = "feature_list"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.FeatureLists.FeatureListEntry"

    sequence = file_descriptor.message_type.add()
    sequence.name = "SequenceExample"
    field = sequence.field.add()
    field.name = "context"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features"
    field = sequence.field.add()
    field.name = "feature_lists"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.FeatureLists"


def _add_map_key_value(entry, value_type_name: str) -> None:
    key = entry.field.add()
    key.name = "key"
    key.number = 1
    key.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    key.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    value = entry.field.add()
    value.name = "value"
    value.number = 2
    value.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    value.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    value.type_name = value_type_name
