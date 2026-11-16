# Logging

Uni-Agent's logging is built on the standard `logging` module and is
configured automatically — to add log records in your code, get a
module-level logger with `getLogger(__name__)` and call it at the level you
need:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("request routed to %s", replica_id)
logger.warning("replica %s overloaded (kv=%.2f)", replica_id, kv_perc)
```

No configuration call is required: the framework wires logging on first use,
and standalone components (for example the router) bootstrap it on import.
Records are formatted uniformly and printed to stdout; components that run
inside a `sample_logging` block are additionally written to that run's log
file when one is configured.

## Session-scoped run logging

Wrap a run with the `sample_logging` context manager (`with` or `async with`)
to bind a log ID to all records emitted inside the block, and optionally route
them to a per-run log file:

```python
from uni_agent.logging import sample_logging

with sample_logging("run-001", log_path="/tmp/run-001.log"):
    run_episode()
```

With `log_path` the run's records are written there; with `None` nothing hits
disk.

## Log level

The log level is resolved in this order:

1. The `UNI_AGENT_LOG_LEVEL` environment variable.
2. `INFO` otherwise.

Accepted values are `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`; an
invalid value falls back to `INFO`. To turn on debug output for a process:

```bash
export UNI_AGENT_LOG_LEVEL=DEBUG
```

The level applies to uni-agent's own `uni_agent.*` loggers; the host
process's root logging configuration is never read or written.
