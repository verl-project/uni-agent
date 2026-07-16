Welcome to Uni-Agent's documentation!
=====================================

Uni-Agent is a framework for building, running, and training long-horizon agent
workflows. It provides persistent sandboxes, tool-based interaction loops, and
`verl` integration for scalable reinforcement learning.

The documentation is organized into three sections:

- **Quickstart** — the main path from zero to training: install Uni-Agent,
  launch a sandbox, build a simple tool-using agent, scale interaction
  across many tasks, and train with fully asynchronous RL.
- **Guides** — deeper walkthroughs and config references for going beyond
  the quickstart.
- **Blog** — long-form posts: design notes, results, and write-ups of new
  features.

.. raw:: html

   <div style="margin: 24px 0; text-align: left;">
     <img src="uni-agent.png" alt="Uni-Agent overview" style="width: 100%; max-width: 700px; height: auto;" />
   </div>

.. _Contents:

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   start/installation.md
   start/quick_start.md
   start/agent_env.md
   start/arxiv_search_agent.md
   start/agent_interaction.md
   start/agent_train.md

.. toctree::
   :maxdepth: 2
   :caption: Guides

   start/terminal_bench_eval.md
   start/search_agent.md

.. toctree::
   :maxdepth: 1
   :caption: Blog

   blog/vision.md
   blog/lark_agent.md
