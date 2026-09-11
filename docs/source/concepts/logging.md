# Logging

Uni-Agent provides two logging setups: process-scoped console logging and
session-scoped run logging. Use the former when a component only needs stdout
output; use the latter when each run needs its own log ID and optional log file.

## Process-scoped: `setup_console_logging`

Call it once at process startup for components without a per-run `LogContext`
(for example a router or CLI entry point):

```python
import logging

from uni_agent.logging import setup_console_logging

setup_console_logging(level=logging.DEBUG)
```

The call is idempotent: calling it again only re-applies the level, so it is
safe to call from library code and from an embedding process.

## Session-scoped: `sample_logging`

Bind a log ID — and optionally a log file — per run with the `sample_logging`
context manager (`with` or `async with`):

```python
from uni_agent.logging import sample_logging

with sample_logging("run-001", log_path="/tmp/run-001.log"):
    run_episode()
```

Records in this block carry the log ID and are also written to `log_path` when
one is given; with `None` nothing hits disk.

The two scopes compose: use `setup_console_logging` for process-wide stdout
control, and `sample_logging` per run when runs need their own log files.

## Log level

The global level is resolved in this order:

1. The explicit `level` argument of `setup_console_logging`, if given.
2. The `UNI_AGENT_LOG_LEVEL` environment variable.
3. `INFO` otherwise.

Accepted values for both are `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.
An invalid value falls back to `INFO`.

```bash
export UNI_AGENT_LOG_LEVEL=DEBUG
python -m my_router --serve
```
