from __future__ import annotations

import json
import struct

import numpy as np

from particle_jepa.data.lts_dataset import (
    LearningToSimulateConfig,
    LearningToSimulateDataset,
    _sequence_example_class,
)


def test_lts_dataset_decodes_sequence_example(tmp_path) -> None:
    metadata = {
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
        "sequence_length": 3,
        "default_connectivity_radius": 0.4,
        "dim": 2,
        "dt": 0.1,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    payload = _fake_lts_sequence_payload()
    _write_tfrecord(tmp_path / "train.tfrecord", [payload])

    dataset = LearningToSimulateDataset(
        LearningToSimulateConfig(
            root=tmp_path,
            future_offset=1,
            sample_stride=1,
            max_samples_per_trajectory=None,
        )
    )
    context, future = dataset[0]

    assert len(dataset) == 3
    assert context.x.shape == (4, 7)
    assert future.x.shape == (4, 7)
    assert context.edge_attr.shape[1] == 6
    assert context.y_acceleration.shape == (4, 2)
    assert context.horizon.item() == 1


def _fake_lts_sequence_payload() -> bytes:
    sequence = _sequence_example_class()()
    positions = np.array(
        [
            [[0.1, 0.1], [0.2, 0.1], [0.8, 0.8], [0.8, 0.7]],
            [[0.11, 0.1], [0.21, 0.1], [0.79, 0.8], [0.79, 0.7]],
            [[0.12, 0.1], [0.22, 0.1], [0.78, 0.8], [0.78, 0.7]],
            [[0.13, 0.1], [0.23, 0.1], [0.77, 0.8], [0.77, 0.7]],
        ],
        dtype=np.float32,
    )
    particle_types = np.array([0, 0, 1, 1], dtype=np.int64)
    sequence.context.feature["particle_type"].bytes_list.value.append(particle_types.tobytes())
    feature_list = sequence.feature_lists.feature_list["position"]
    for frame in positions:
        feature = feature_list.feature.add()
        feature.bytes_list.value.append(frame.reshape(-1).tobytes())
    return sequence.SerializeToString()


def _write_tfrecord(path, records: list[bytes]) -> None:
    with path.open("wb") as handle:
        for record in records:
            handle.write(struct.pack("<Q", len(record)))
            handle.write(b"\x00\x00\x00\x00")
            handle.write(record)
            handle.write(b"\x00\x00\x00\x00")
