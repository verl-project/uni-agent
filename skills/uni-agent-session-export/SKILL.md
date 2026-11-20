---
name: uni-agent-session-export
description: Diagnose missing or conflicting trajectory counts in local uni-agent Gateway debug launcher exports. Use when identifying which export belongs to a reported request, particularly when several batches contain the same session ID.
---

# Inspect a Gateway session export

Start with the original ticket and archive. Identify the **export directory** as well as the session ID before attributing a count mismatch. A directory named after the right session may belong to another invocation or be a copy of the same export.

From the repository root, run the read-only indexer (Python 3.10+):

```sh
python skills/uni-agent-session-export/scripts/inspect_session.py /path/to/archive
```

When using an installed skill, invoke `scripts/inspect_session.py` relative to this file. Add `--match 'literal from the ticket'` to locate a request marker. The search reports matching JSON fields and JSONL line numbers, not conversation text. Look at the referenced original: a substring of a longer ID or an assistant echo is not evidence of ownership. Other evidence, such as plain text launcher logs, requires separate inspection.

**Resolve the batch.** For a repeated ID, keep each relative directory as a distinct candidate. Link the reported request to a candidate through an independently attributable request ID, invocation log, or run manifest. A timestamp helps only when the ticket has a compatible time window and the two runs fall on different sides of it. Directory names, copy times, or which export looks broken do not establish ownership. Identical bytes can be duplicate copies; a different hash alone does not prove a separate invocation. If both candidates still fit, report both and request the specific missing fact that separates them. If no candidate fits, identify the expected run or archive that is missing.

**Reconcile within each candidate.** Compare `session_metadata.json:trajectories_count` with the number of parseable JSON object lines in its sibling `trajectories.jsonl`. Verify the row IDs against the metadata and snapshot; equal counts with mixed IDs do not clear an export. Report missing IDs or invalid lines as such. A missing file is unknown; an existing empty file contains zero objects. Blank lines do not count. The snapshot is taken before finalization; its `session_state.num_trajectories` is not the final count. Archived metadata may contain paths from the original host; always inspect files in the same export directory.

**Return a diagnosis someone can verify.** Cite the relative directory, the evidence linking it to the ticket (or why that link is unresolved), the declared and observed counts, and the smallest next check that could change the conclusion. Keep an observed discrepancy separate from its cause: these files alone cannot prove that finalization failed, that rows were lost, or that training consumed them.
