import time

import pytest

from benchcaddy.isolation.process import run_isolated


def test_worker_timeout_path_raises_timeout_error():
    with pytest.raises(TimeoutError):
        run_isolated(time.sleep, args=(2,), fresh_process=True, timeout=0.1)
