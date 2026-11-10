# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded allowlisted queries; never accept caller-supplied PromQL or backend URLs."""

import json
import asyncio
import math
import re

import aiohttp

from .telemetry_contract import CollectionConfig


METRICS = frozenset(("gpu_utilization_percent", "memory_utilization_percent", "memory_used_bytes",
                     "memory_total_bytes", "temperature_c", "power_watts", "sm_active_ratio",
                     "sm_occupancy_ratio", "tensor_active_ratio", "dram_active_ratio"))


def prepare_query(state, params, now):
    allowed = {"metric", "mode", "node_id", "pool", "gpu_uuid", "start", "end", "step"}
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError("unknown metric query parameter")
    metric, mode = params.get("metric"), params.get("mode", "latest")
    if metric not in METRICS or mode not in ("latest", "history"):
        raise ValueError("invalid metric or query mode")
    telemetry = state.get("telemetry", {})
    config = CollectionConfig.parse(telemetry.get("default", {}))
    pool = params.get("pool")
    if pool is not None:
        pool = next((key for key, record in state["pools"].items() if pool in (key, record["name"])), None)
        if pool is None:
            raise ValueError("unknown pool")
    node_id = params.get("node_id")
    if node_id is not None and node_id not in state.get("nodes", {}):
        raise ValueError("unknown node")
    nodes = [node for key, node in state.get("nodes", {}).items()
             if (node_id is None or key == node_id) and (pool is None or node["pool"] == pool)]
    gpus = {gpu for node in nodes for gpu in node.get("effective_resources", node["resources"])["gpu_uuids"]}
    if "gpu_uuid" in params:
        if params["gpu_uuid"] not in gpus:
            raise ValueError("GPU not in requested resource scope")
        gpus = {params["gpu_uuid"]}
    if len(gpus) > 1000:
        raise ValueError("metric query scope exceeds limit")
    service = telemetry.get("bindings", {}).get(node_id)
    if service is not None:
        config = CollectionConfig.parse(telemetry.get("overrides", {}).get(service, telemetry.get("default", {})))
    name = ('cape_' if metric.startswith('gpu_') else 'cape_gpu_') + metric
    selector = name + '{scheduler_id=' + json.dumps(state["scheduler_id"])
    # UUIDs originate in validated enrolled resources; still escape the regex and string layers.
    selector += ',gpu_uuid=~' + json.dumps("|".join(re.escape(gpu) for gpu in sorted(gpus)))
    if service is not None:
        selector += ',service_id=' + json.dumps(service)
    selector += '}'
    freshness = 'cape_gpu_sample_expires_at_seconds' + selector[selector.index('{'):]
    query = '({}) and on(scheduler_id,service_id,gpu_uuid) ({} > time())'.format(selector, freshness)
    options = {"query": query, "timeout": "5s"}
    if mode == "history":
        try:
            start, end, step = (float(params[key]) for key in ("start", "end", "step"))
        except (ValueError, TypeError, KeyError):
            raise ValueError("history requires numeric start/end/step") from None
        if (not all(math.isfinite(v) for v in (start, end, step)) or start < 0 or end < start
                or end > now + 60 or end - start > 31 * 86400 or step < 1
                or (math.floor((end - start) / step) + 1) * max(len(gpus), 1) > 100000):
            raise ValueError("metric query range exceeds limit")
        options.update(start=str(start), end=str(end), step=str(step))
    elif any(key in params for key in ("start", "end", "step")):
        raise ValueError("time range requires history mode")
    return {"path": "/api/v1/query_range" if mode == "history" else "/api/v1/query", "params": options,
            "mode": mode, "metric": metric, "gpus": sorted(gpus), "scheduler_id": state["scheduler_id"],
            "enabled": config.enabled, "service_id": service}


def sanitize_result(plan, body):
    expected = "matrix" if plan["mode"] == "history" else "vector"
    if (not isinstance(body, dict) or body.get("status") != "success" or not isinstance(body.get("data"), dict)
            or body["data"].get("resultType") != expected):
        raise ValueError("invalid metric backend response")
    result = body["data"].get("result")
    if not isinstance(result, list) or len(result) > 1000:
        raise ValueError("metric backend result limit exceeded")
    sanitized, count = [], 0
    for item in result:
        if not isinstance(item, dict) or not isinstance(item.get("metric"), dict):
            raise ValueError("invalid metric labels")
        labels = item.get("metric", {})
        if (labels.get("gpu_uuid") not in plan["gpus"] or labels.get("scheduler_id") != plan["scheduler_id"]
                or (plan["service_id"] is not None and labels.get("service_id") != plan["service_id"])):
            raise ValueError("metric backend scope mismatch")
        if labels.get("service_id") is not None and not isinstance(labels["service_id"], str):
            raise ValueError("invalid metric service label")
        points = item.get("values") if expected == "matrix" else [item.get("value")]
        if not isinstance(points, list):
            raise ValueError("invalid metric backend points")
        clean = []
        for point in points:
            count += 1
            if count > 100000 or not isinstance(point, list) or len(point) != 2:
                raise ValueError("invalid metric backend points")
            if type(point[0]) not in (int, float) or not isinstance(point[1], str):
                raise ValueError("invalid metric backend point types")
            timestamp, value = float(point[0]), float(point[1])
            if not math.isfinite(timestamp):
                raise ValueError("invalid metric timestamp")
            clean.append([timestamp, value if math.isfinite(value) else None])
        sanitized.append({"gpu_uuid": labels["gpu_uuid"], "service_id": labels.get("service_id"), "samples": clean})
    return {"state": "ready" if sanitized else "missing", "metric": plan["metric"], "mode": plan["mode"], "series": sanitized}


async def fetch_query(plan, base_url):
    disabled = not plan["enabled"] and plan["mode"] == "latest"
    empty = {"state": "disabled" if disabled else "unavailable", "metric": plan["metric"], "mode": plan["mode"], "series": []}
    if disabled or not base_url:
        return empty
    if not plan["gpus"]:
        return dict(empty, state="missing")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7)) as session:
            async with session.get(base_url.rstrip("/") + plan["path"], params=plan["params"], allow_redirects=False) as response:
                if response.status != 200:
                    return empty
                chunks, length = [], 0
                async for chunk in response.content.iter_chunked(65536):
                    length += len(chunk)
                    if length > 8 * 1024 * 1024:
                        return empty
                    chunks.append(chunk)
                return sanitize_result(plan, json.loads(b"".join(chunks)))
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError, KeyError, AttributeError):
        return empty
