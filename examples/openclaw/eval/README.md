# OpenClaw-Test: End-to-End Evaluation for OpenClaw-RL Training Methods

This directory contains an automated evaluation suite that tests the **real-world effectiveness** of models trained with OpenClaw-RL method.

## What Does This Test Do?

The evaluation simulates a realistic multi-turn agentic workflow using **GSM8K math problems** as the task domain. An external LLM (the "user") interacts with the OpenClaw agent (your trained model) through the OpenClaw gateway API, testing whether the agent can:

- Read files from its workspace
- Solve math problems with complete step-by-step reasoning
- Follow stylistic instructions (e.g., rewrite in a more natural tone)
- Write results back to files
- Grade existing solutions against ground truth
- Produce detailed, friendly feedback

The test consists of **three sequential phases** (you can also run them together, but you need to obtain homework1 and homework2 first if you want to try this joint optimization):

### Phase 1: Student Chat (`student_chat.py`)

An external LLM role-plays as a **lazy student** who asks the OpenClaw agent to do their homework. For each GSM8K problem:

1. The problem is written to `homework/i.txt` in the OpenClaw workspace.
2. The "student" asks the agent to read the file and solve it.
3. If the agent's answer looks too AI-like (bold text, numbered lists, etc.), the student tells it to rewrite in a more natural style.
4. Once satisfied, the student asks the agent to append the answer to the homework file.
5. The student says `HOMEWORK_DONE` to end the session.

This phase tests the agent's **instruction following**, **math reasoning**, **file I/O**, and **style adaptation** abilities.

### Phase 2: TA Chat (`TA_chat.py`)

An external LLM role-plays as a **TA** who grades the student's submissions. For each problem:

1. If needed, `homework/` is copied to `homework1/` in the OpenClaw workspace.
2. The TA provides the original question and ground truth answer to the agent.
3. The agent reads the student's submission from `homework1/i.txt`, compares it with the correct answer, and writes grading comments.
4. If the comments are too brief or not specific enough, the TA asks for a rewrite.
5. Once satisfied, the TA asks the agent to append the comments to the file.
6. The TA says `GRADING_DONE` to end the session.

This phase tests the agent's **reading comprehension**, **evaluation accuracy**, **feedback specificity**, and **multi-step file operations**.

### Phase 3: Teacher Chat (`teacher_chat.py`)

An external LLM role-plays as a **teacher** who reviews the already graded homework and writes comments about the student's strengths and weaknesses. For each problem:

1. If needed, `homework1/` is copied to `homework2/` in the OpenClaw workspace.
2. The teacher provides the original question and ground truth answer to the agent.
3. The agent reads the graded submission from `homework2/i.txt` and writes friendly, patient feedback about strengths and weaknesses.
4. If the comments are not friendly or patient enough, the teacher asks for a rewrite.
5. Once satisfied, the teacher asks the agent to append the comments to the file.
6. The teacher says `COMMENT_DONE` to end the session.

This phase tests the agent's **review quality**, **tone control**, **supportive feedback**, and **multi-step file operations**.

> **Run order matters:** Run `student_chat.py` first so the homework files contain student solutions, then run `TA_chat.py` to grade them, then run `teacher_chat.py` to add teacher comments.

---

## Architecture Overview

```
┌─────────────────────┐         ┌───────────────────────────────┐
│   External LLM      │         │      OpenClaw RL Server       │
│ (Student/TA/Teacher) │         │   (your trained model)        │
│  Port 30001         │         │   Port 30000                  │
│  via launch_user_   │         │   via openclaw-rl/opd/combine │
│  llm.sh or closed-  │         │   shell scripts               │
│  source API          │         │                               │
└────────┬────────────┘         └──────────┬────────────────────┘
         │                                 │
         │ student/TA/teacher messages      │  agent responses
         │                                 │
         └──────────┐     ┌────────────────┘
                    ▼     ▼
              ┌──────────────────┐
              │  student_chat.py │
              │  TA_chat.py      │
              │  teacher_chat.py │
              │  (orchestrator)  │
              └──────────────────┘
```

---

## Step-by-Step Guide

### Prerequisites

- A running OpenClaw environment (see the [main README](../README.md))
- Python 3.12 with `requests` and `openai` packages installed
- A `GSM8K.json` dataset file (JSON array with `question` and `ground_truth_answer` fields per entry)
- 4 available GPUs for `run_openclaw_rl_eval_4gpu.sh`: GPU 0,1,2 are used by the RL service, and GPU 3 is used by the external role LLM.

### Step 1: Configure the 4-GPU Evaluation Script

Edit the configurable settings at the top of `run_openclaw_rl_eval_4gpu.sh`:

```bash
ROOT="/path/frameworks/uni-agent"
MODEL_PATH="/path/models/Qwen3-VL-4B-Instruct"
DATASET="${ROOT}/examples/openclaw/eval/GSM8K.json"
```

The script uses the same model path for both the RL service and the user LLM by default:

- RL service endpoint: `http://127.0.0.1:30000`
- user LLM endpoint: `http://127.0.0.1:30001/v1`
- RL API key: `rl-local-token`
- user LLM API key: `user-llm-local-token`
- user LLM model name: `qwen3-vl-4b-user-llm`

You can override the evaluation scale without editing the script:

```bash
export NUM_PROBLEMS=36
export MAX_TURNS=8
export TOTAL_ROLLOUT_STEPS=2000
```

For long conversations, the script also bounds the prompt/reply text sent through the driver:

```bash
export OPENCLAW_DRIVER_HISTORY_MAX_CHARS=20000
export OPENCLAW_AGENT_REPLY_MAX_CHARS=12000
```

### Step 2: Run the Full Pipeline

Run the script from the repository root:

```bash
cd /path/frameworks/uni-agent
bash examples/openclaw/eval/run_openclaw_rl_eval_4gpu.sh
```

The script performs the complete evaluation flow:

1. Cleans up stale `uni_agent.openclaw.rl.train_entry`, `vllm serve`, and Ray processes.
2. Starts the RL service on GPU 0,1,2 by running `examples/openclaw/train_rl.sh`.
3. Waits for `http://127.0.0.1:30000/healthz` to become ready.
4. Starts the user LLM on GPU 3 by running `examples/openclaw/eval/launch_user_llm.sh`.
5. Waits for `http://127.0.0.1:30001/v1/models` to become ready.
6. Exports the OpenClaw gateway and OpenAI-compatible environment variables used by the eval scripts.
7. Runs `student_chat.py`, then `TA_chat.py`, then `teacher_chat.py`.

### Step 3: Check Logs and Outputs

Runtime logs are written under the repository-level `logs/` directory:

- RL service log: `logs/rl_eval_rl_server.log`
- user LLM log: `logs/rl_eval_user_llm.log`

Evaluation outputs are written in this directory:

- Student stage: `results_student.txt`
- TA stage: `results_TA.txt`
- Teacher stage: `results_teacher.txt`

The workspace defaults to `~/.openclaw/workspace`:

- `homework/`: student solutions
- `homework1/`: TA-graded submissions
- `homework2/`: teacher-commented submissions

### Equivalent Manual Eval Commands

After the RL service and user LLM are ready, `run_openclaw_rl_eval_4gpu.sh` runs the three stages with these environment variables:

```bash
export OPENCLAW_GATEWAY_URL="http://127.0.0.1:30000"
export OPENCLAW_GATEWAY_TOKEN="rl-local-token"
export OPENCLAW_ENDPOINT_MODE="gateway"
export OPENCLAW_AGENT_MODEL="default"
export OPENCLAW_WORKSPACE="${HOME}/.openclaw/workspace"

export OPENAI_BASE_URL="http://127.0.0.1:30001/v1"
export OPENAI_API_KEY="user-llm-local-token"
export EXTERNAL_MODEL="qwen3-vl-4b-user-llm"
export OPENCLAW_DRIVER_HISTORY_MAX_CHARS=20000
export OPENCLAW_AGENT_REPLY_MAX_CHARS=12000

python student_chat.py --dataset "${DATASET}" --num-problems "${NUM_PROBLEMS}" --max-turns "${MAX_TURNS}"
python TA_chat.py --dataset "${DATASET}" --num-problems "${NUM_PROBLEMS}" --max-turns "${MAX_TURNS}"
python teacher_chat.py \
    --dataset "${DATASET}" \
    --num-problems "${NUM_PROBLEMS}" \
    --max-turns "${MAX_TURNS}"
```

---

## Command-Line Arguments

`student_chat.py`, `TA_chat.py`, and `teacher_chat.py` accept the same arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to the GSM8K JSON file |
| `--num-problems` | `5` | Number of problems to process |
| `--max-turns` | `8` | Maximum conversation turns per problem |
| `--max-retries` | `3` | Maximum retries per network call |
| `--output` | See below | Output file for the first OpenClaw reply from each problem |

Default output files:

| Script | Default output |
|---|---|
| `student_chat.py` | `results_student.txt` |
| `TA_chat.py` | `results_TA.txt` |
| `teacher_chat.py` | `results_teacher.txt` |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENCLAW_GATEWAY_TOKEN` | Yes | — | Auth token for the OpenClaw gateway |
| `OPENAI_API_KEY` | Yes | — | API key for the external LLM (student/TA/teacher) |
| `OPENCLAW_GATEWAY_URL` | No | `http://localhost:18789` | OpenClaw gateway base URL. `run_openclaw_rl_eval_4gpu.sh` sets this to `http://127.0.0.1:30000`. |
| `OPENCLAW_ENDPOINT_MODE` | No | `gateway` | OpenClaw endpoint mode used by the eval scripts |
| `OPENCLAW_AGENT_MODEL` | No | `default` | Model name sent to the OpenClaw gateway |
| `OPENCLAW_WORKSPACE` | No | `~/.openclaw/workspace` | Path to the OpenClaw workspace directory |
| `OPENAI_BASE_URL` | No | *(OpenAI default)* | Base URL for the external LLM API. `run_openclaw_rl_eval_4gpu.sh` sets this to `http://127.0.0.1:30001/v1`. |
| `EXTERNAL_MODEL` | No | `gpt-4o` | Model name for the external LLM. `run_openclaw_rl_eval_4gpu.sh` sets this to `qwen3-vl-4b-user-llm`. |
| `OPENCLAW_DRIVER_HISTORY_MAX_CHARS` | No | `20000` in `run_openclaw_rl_eval_4gpu.sh` | Maximum conversation-history characters sent to the external role LLM |
| `OPENCLAW_AGENT_REPLY_MAX_CHARS` | No | `12000` in `run_openclaw_rl_eval_4gpu.sh` | Maximum agent-reply characters forwarded to the external role LLM |

---



## File Structure

```
eval/
├── README.md              # This file
├── run_openclaw_rl_eval_4gpu.sh
│                           # 4-GPU end-to-end RL eval pipeline
├── launch_user_llm.sh     # Script to host the external LLM via vLLM
├── student_chat.py        # Phase 1: Student asks agent to solve homework
├── TA_chat.py             # Phase 2: TA asks agent to grade homework
├── teacher_chat.py        # Phase 3: Teacher asks agent to comment on strengths and weaknesses
└── GSM8K.json             # Dataset (to be placed here)
```
