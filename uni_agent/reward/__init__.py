from .registry import load_reward_spec

_LAZY_EXPORTS = {
    "SearchRewardSpec": ".search",
    "SWEBenchRewardSpec": ".swe_bench",
    "SWEBenchMultilingualRewardSpec": ".swe_bench_multilingual",
    "R2EGymRewardSpec": ".r2e_gym",
    "SWEREBenchRewardSpec": ".swe_rebench",
    "SWEREBenchV2RewardSpec": ".swe_rebench_v2",
}

__all__ = [
    "load_reward_spec",
    "SearchRewardSpec",
    "SWEBenchRewardSpec",
    "SWEBenchMultilingualRewardSpec",
    "R2EGymRewardSpec",
    "SWEREBenchRewardSpec",
    "SWEREBenchV2RewardSpec",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module = import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
