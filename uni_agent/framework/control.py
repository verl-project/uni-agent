"""Stop the unfinished portion of one validation and wait for its committed metrics."""

import argparse
import json
import time

import ray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--actor", required=True, help="Configured agent_framework.control_actor_name")
    parser.add_argument("--validation-id", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    ray.init(address=args.address, namespace=args.namespace)
    try:
        worker = ray.get_actor(args.actor)
        ray.get(worker.stop_validation.remote(args.validation_id), timeout=args.timeout)
        deadline = time.monotonic() + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Validation stop requested, but metrics have not been committed")
            state = ray.get(worker.validation_status.remote(args.validation_id), timeout=remaining)
            if state["phase"] == "finished":
                print(json.dumps(state, allow_nan=False), flush=True)
                return
            if state["phase"] == "unsafe":
                print(json.dumps(state, allow_nan=False), flush=True)
                raise RuntimeError("Validation metrics saved, but generation cleanup failed; training remains blocked")
            time.sleep(min(0.1, remaining))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
