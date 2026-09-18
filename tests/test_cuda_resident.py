import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOCudaError


@pytest.mark.skipif(not eo.cuda.is_available(), reason="CUDA extension/device unavailable")
def test_resident_chain_matches_numpy_without_intermediate_downloads():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = np.eye(4, dtype=np.float32)
    da, db = eo.cuda.to_device(a), eo.cuda.to_device(b)
    result = ((da @ db) + da).relu()
    assert result.device == "cuda"
    np.testing.assert_allclose(result.numpy(), np.maximum((a @ b) + a, 0), rtol=1e-5)
    assert result.sum() == pytest.approx(float(np.maximum((a @ b) + a, 0).sum()))


@pytest.mark.skipif(eo.cuda.is_available(), reason="CUDA is available")
def test_upload_fails_clearly_without_cuda():
    with pytest.raises(EveryOCudaError, match="CUDA extension"):
        eo.cuda.to_device(np.ones(4, dtype=np.float32))
