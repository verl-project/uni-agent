# Search Agent Training Example

This directory contains the ASearcher + LocalWiki search-agent training example.

For the full setup guide, see the documentation:

[Train a Search Agent](https://uni-agent.readthedocs.io/en/latest/start/search_agent.html)

## Files

- `agent_config.yaml`: agent loop config for the search agent.
- `runtime_env.yaml`: Ray runtime env for training.
- `run_localwiki_and_train.sh`: starts LocalWiki and submits training.
- `train_fully_async_128K.sh`: fully async training launcher.

Minimal entry point:

```bash
DATA_ROOT=/path/to/data_root \
bash examples/search_agent/run_localwiki_and_train.sh
```
