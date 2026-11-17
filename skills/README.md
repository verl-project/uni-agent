# Contributing a troubleshooting skill

Keep one source under `skills/<name>/`: a short `SKILL.md` for the investigative decisions, plus `scripts/` for reproducible checks that engineers can also run directly. This layout follows the [AI Infra Auto Driven Skills collection](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills). Nothing here changes the Gateway runtime or requires automatic skill installation.

| Skill | Start here | Standalone tool |
| --- | --- | --- |
| Gateway session export | [Select the right export and check its records](uni-agent-session-export/SKILL.md) | [Inspect local exports](uni-agent-session-export/scripts/inspect_session.py) |

When changing the export schema or its naming, update the checker, its boundary tests and any affected guidance in the same change. New skills should describe when *not* to use them and what evidence could change a diagnosis. Keep incident archives, private identifiers and answer keys out of the repository.

Use the [shared evaluation entry point](../evals/skills/README.md) to compare the complete package with ordinary tools and with its scripts alone. Do not recommend installation based on script-call counts. Report failures and negative results alongside improvements.
