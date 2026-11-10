#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize actual CUDA event duration separately from per-GPU lock waiting."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def summarize(report, cache_dir):
    metadata = json.loads(Path(str(report) + '.meta.json').read_text())
    run_id = metadata['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
        raise ValueError('invalid run identity')
    rows = []
    for index in range(1, int(metadata['workloads']) + 1):
        text = (cache_dir / 'timing' / run_id / (str(index) + '.txt')).read_text()
        lock = re.search(r'gpu_lock enabled=(\d) wait_ms=([\d.]+)', text)
        device = re.search(r'device_ms=([\d.]+)', text)
        geometry = re.search(r'gpu_sleep launch (.+)', text)
        if not lock or not geometry:
            raise ValueError('missing GPU measurement for command ' + str(index))
        rows.append({'index': index, 'locked': lock[1] == '1',
                     'lock_wait_ms': float(lock[2]),
                     'kernel_ms': float(device[1]) if device else None,
                     'geometry': geometry[1]})

    def stats(key):
        values = sorted(r[key] for r in rows if r[key] is not None)
        return None if not values else {'count': len(values), 'median': statistics.median(values),
                                       'p95': values[math.ceil(.95 * len(values)) - 1], 'max': max(values)}

    result = {'run_id': run_id, 'kernel_ms': stats('kernel_ms'),
              'lock_wait_ms': stats('lock_wait_ms'), 'commands': rows,
              'notes': ['kernel_ms uses CUDA events around the fixed-work vecadd kernel only.',
                        'Existing report gpu_time includes lock waiting and CUDA initialization.',
                        'Compare equal workload/geometry, GPU, pool slots and warm-cache conditions.',
                        'Cooperative locking cannot exclude nonparticipating GPU workloads.']}
    output = Path(str(report) + '.gpu-timing.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('CUDA kernel duration (ms): ' + json.dumps(result['kernel_ms']))
    print('GPU lock waiting (ms): ' + json.dumps(result['lock_wait_ms']))
    print('GPU timing details: ' + str(output))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.report, args.cache_dir)
