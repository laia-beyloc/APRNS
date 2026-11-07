"""Tests de la memòria de repetició."""
import numpy as np

from ReplayBuffer import ReplayBuffer


def _fill(buffer, n):
    for i in range(n):
        buffer.put(np.full(8, i), np.full(2, i / 10), float(i), np.full(8, i + 1), i % 2 == 0, False)


def test_circular_overwrites_oldest():
    buffer = ReplayBuffer(3)
    _fill(buffer, 5)
    assert len(buffer) == 3
    assert sorted(buffer.arrays[2].tolist()) == [2.0, 3.0, 4.0]


def test_get_shapes_and_flags():
    buffer = ReplayBuffer(100)
    _fill(buffer, 10)
    s, a, r, s2, term, trunc = buffer.get(4)
    assert s.shape == (4, 8) and a.shape == (4, 2) and r.shape == (4,) and s2.shape == (4, 8)
    assert term.shape == trunc.shape == (4,)
    # Coherència de cada fila: s' = s + 1 i terminated = (i parell).
    np.testing.assert_array_equal(s2[:, 0], s[:, 0] + 1)
    np.testing.assert_array_equal(term, (r % 2 == 0).astype(np.float32))


def test_state_dict_roundtrip():
    original = ReplayBuffer(50)
    _fill(original, 20)
    restored = ReplayBuffer(1)
    restored.load_state_dict(original.state_dict())
    np.random.seed(0)
    batch_a = original.get(8)
    np.random.seed(0)
    batch_b = restored.get(8)
    for x, y in zip(batch_a, batch_b):
        np.testing.assert_array_equal(x, y)
    assert (restored.capacity, len(restored), restored.position) == (50, 20, 20)
