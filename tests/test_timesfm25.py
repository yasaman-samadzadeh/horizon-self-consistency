"""TimesFM 2.5 wrapper. The model checks run only where timesfm is installed (uv sync --extra timesfm):

    python -m pytest -q tests/test_timesfm25.py
"""

import numpy as np
import pytest

from hsc.audit.windows import load_windows
from hsc.common.timesfm25 import to_levels


def test_to_levels_drops_the_mean_channel():
    out = np.broadcast_to(np.arange(10.0), (2, 3, 10))
    assert np.array_equal(to_levels(out), np.broadcast_to(np.arange(1.0, 10.0), (2, 3, 9)))


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("timesfm")
    from hsc.common.timesfm25 import TimesFM25

    return {"base": TimesFM25(), "ar": TimesFM25(max_horizon=256), "long": TimesFM25(max_context=1024)}


def test_horizons_up_to_128_are_one_direct_patch(models):
    contexts, _ = load_windows("ar2", 4, 20261104)
    ext = np.concatenate([contexts, contexts[:, -1:]], axis=1)  # 501 points, as in the composite
    for c in (contexts, ext):
        q64 = models["base"].forecast(c, 64)
        assert q64.shape == (4, 64, 9) and np.isfinite(q64).all()
        assert (np.diff(q64, axis=-1) >= 0).all()
        # the requested horizon only truncates the one output patch
        assert np.allclose(q64, models["base"].forecast(c, 128)[:, :64], rtol=0, atol=1e-5)
        # with max_horizon 256 the second patch is decoded autoregressively; the first is unchanged
        assert np.allclose(models["ar"].forecast(c, 256)[:, :128], models["base"].forecast(c, 128), atol=1e-4)
        # extra left padding is masked out of attention
        assert np.allclose(models["long"].forecast(c, 64), q64, atol=1e-4)
