"""Smoke test for the validity simulation script (small and fast)."""

from tbshift.experiments.validity_simulation import simulate


def test_simulation_shape_and_floor_behaviour():
    df = simulate(n_pos_grid=(5, 50), repeats=200, seed=1)
    assert len(df) == 4
    assert set(df["method"]) == {"marginal", "certified"}
    # Below both label floors every threshold is trivial: flag everyone.
    below = df[df["n_pos"] == 5]
    assert (below["trivial_rate"] == 1.0).all()
    assert (below["mean_sensitivity"] == 1.0).all()
    # With 50 positives neither method should be trivial.
    above = df[df["n_pos"] == 50]
    assert (above["trivial_rate"] == 0.0).all()


def test_simulation_is_reproducible():
    a = simulate(n_pos_grid=(30,), repeats=100, seed=5)
    b = simulate(n_pos_grid=(30,), repeats=100, seed=5)
    assert a.equals(b)
