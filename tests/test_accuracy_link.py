import numpy as np
import pytest

from hsc.audit.accuracy_link import by_model
from hsc.common.paths import RESULTS
from hsc.common.quantiles import band_w1


def test_triangle_inequality_on_quantile_vectors():
    rng = np.random.default_rng(0)
    a, b, c = (np.sort(rng.normal(size=(50, 9)), axis=-1) for _ in range(3))
    assert np.all(band_w1(a, b) <= band_w1(a, c) + band_w1(c, b) + 1e-12)


@pytest.mark.skipif(not (RESULTS / "main" / "ar1" / "oracle.npz").exists(),
                    reason="needs the stored arrays of the main run (results/main/*/*.npz)")
def test_oracle_has_zero_error_and_defect_equals_composed_error():
    rows = by_model("ar1")
    oracle = rows["oracle"]
    assert max(oracle["E_dir"]["mean"]) < 1e-12
    assert np.allclose(oracle["D"]["mean"], oracle["E_comp"]["mean"])
    for name, r in rows.items():
        assert all(s <= 1 + 1e-9 for s in r["share"]), name
