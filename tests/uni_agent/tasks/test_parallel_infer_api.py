import pytest

from examples.inference.parallel_infer_api import _allocate_worker_concurrency


@pytest.mark.parametrize(
    ("total_concurrency", "num_workers", "expected"),
    [
        (1, 1, [1]),
        (8, 4, [2, 2, 2, 2]),
        (10, 3, [4, 3, 3]),
    ],
)
def test_allocate_worker_concurrency_preserves_global_limit(total_concurrency, num_workers, expected):
    limits = _allocate_worker_concurrency(total_concurrency, num_workers)

    assert limits == expected
    assert sum(limits) == total_concurrency
