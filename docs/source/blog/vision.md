# Vision: Agents That Grow With You

*2026-05-25 · [Yuyang Ding](https://yyding1.github.io/)*

**This is a proposal, not a release announcement.**

Below is the research direction we want Uni-Agent to enable, the two flagship agents we are framing the work around.

## Project Milo and Miko

We frame the work around two flagship agents: one for the human side of work, one for the engineering side.

**🧠 Project Milo: An chat agent that actually gets you.** Reads intent and subtext, learns what matters to you over time, and on top of that helps you get work done across schedules, mail, and docs.

**💻 Project Miko: An coding agent that actually gets the problem.** Reads specs and codebases, reasons through real engineering challenges, and on top of that manages the whole project for excellent end-to-end performance.

## Online Reinforcement Learning

The bigger bet behind both agents: **once an agent lives next to a user and uses real tools, every conversation is a training signal.** Turning that signal into a model that keeps improving is hard on both the infrastructure and algorithm side.

**Infrastructure**

- **RL training as a service.** Today's RL stacks are built for one-shot research runs. RL in a product needs a long-lived pipeline that continuously ingests trajectories, schedules updates, and rotates fresh checkpoints back to serving.
- **Agent gateway.** A single endpoint that any OpenAI-compatible agent calls without modification, recording full token-level trajectories with consistent tokenization and low latency ([verl RFC #5790](https://github.com/verl-project/verl/issues/5790), [PR #25](https://github.com/verl-project/uni-agent/pull/25)).

**Algorithm**

- **Cleaning noisy user data.** Real conversations contain PII, off-topic chatter, and heavy-user bias. Filtering pipelines must strip the junk without losing the high-signal trajectories that drive learning.
- **Modeling user intent as reward.** Explicit feedback is sparse. Implicit signals like edits and retention are noisy and easy to game into sycophancy. Designing a reward model that captures what users actually want is its own subproject.
