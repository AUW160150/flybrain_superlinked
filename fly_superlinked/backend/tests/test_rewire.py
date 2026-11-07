import numpy as np

from backend.brain.circuit import circuit_edges, degree_preserving_rewire, load_circuit


def test_rewired_preserves_in_and_out_degree_exactly():
    c = load_circuit()
    pre, post, w = circuit_edges("real")
    rp, rq, rw = circuit_edges("rewired")
    N = c["N"]
    assert len(rp) == len(pre)
    assert np.array_equal(np.bincount(pre, minlength=N), np.bincount(rp, minlength=N))
    assert np.array_equal(np.bincount(post, minlength=N), np.bincount(rq, minlength=N))
    # each neuron keeps its outgoing weights (signs and synapse counts) as a multiset
    for i in np.random.default_rng(0).choice(N, 50, replace=False):
        assert np.array_equal(np.sort(w[pre == i]), np.sort(rw[rp == i]))


def test_rewired_is_a_real_shuffle_without_self_loops_or_duplicates():
    pre, post, _ = circuit_edges("real")
    rp, rq, _ = circuit_edges("rewired")
    assert not np.any(rp == rq)
    key = rp.astype(np.int64) * 100000 + rq
    assert len(np.unique(key)) == len(key)
    kept = len(np.intersect1d(pre.astype(np.int64) * 100000 + post, key))
    assert kept / len(key) < 0.05


def test_rewire_is_deterministic_for_a_seed():
    pre = np.array([0, 0, 1, 2, 3, 3, 4]); post = np.array([1, 2, 3, 4, 0, 1, 2]); w = np.arange(7.0)
    a = degree_preserving_rewire(pre, post, w, seed=7)
    b = degree_preserving_rewire(pre, post, w, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
