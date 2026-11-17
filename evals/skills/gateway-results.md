# Gateway session export: evaluation record

The first comparison used three full directories produced by a real Gateway debug launcher with a fake backend and controlled interruptions. It did not use three historical production tickets. Each arm ran three fresh sessions per case, using the same case inputs and search tools. The script was identical wherever provided. Each case contained a unique searchable marker. These measurements concern an earlier frozen version of the guide and checker, not the current revision.

| Case (declared / observed records) | Baseline | Scripts only | Guide + scripts |
| --- | ---: | ---: | ---: |
| Interrupted row write (3 / 0) | 43.1 s | 39.5 s | 47.9 s |
| Metadata still from earlier export (1 / 3) | 48.3 s | 53.9 s | 44.1 s |
| Repeated ID, target batch intact (2 / 2) | 43.8 s | 36.9 s | 56.0 s |

Cells show the median time to the **first complete correct evidence submission**, including startup. This is different from final answer latency in the generic runner. All three arms produced all required facts in 9/9 trials. The guide met its predeclared speed criterion in 0/3 cases (at least 15% faster than each control in two cases). Script use increased from 5/9 to 9/9 when the guide was present; this is a process change, not evidence of higher diagnostic quality. Do not infer developer hours saved. The raw incident-like archives and transcripts are not distributed here; the table is an aggregate record, not a publicly replayable dataset.

## Held-out historical evidence

Admission requires a previously unused ticket, the original contemporaneous materials, and evidence that the same session ID occurred in genuinely distinct invocations. Repeated copies in release or staging archives are not distinct runs. No new incident meeting these criteria was recovered in the reviewed materials. The historical comparison therefore has **zero trials**: neither misidentification rate nor next-action quality can be estimated.

For a future admitted case, evaluate wrong confident identification and factual accuracy separately from whether the proposed next action distinguishes candidates. Preserve every arm's failures and the insufficient-evidence verdict. Do not change the release criterion after seeing answers. Until evidence shows an independent benefit over the scripts-only arm, keep the guide opt-in.

## Runner calibration on a previously used case

Replaying the intact 2/2 case through the new diagnosis runner in three fresh model sessions (one per arm) produced the correct batch and both counts in all arms, with zero wrong identifications. Two reviewers independently scored all three anonymous next actions 2/2: each answer correctly stopped after reconciling the target export. This case does not test asking for more evidence when a batch cannot be identified. Final answers arrived after 55.1, 53.1 and 41.7 seconds (baseline, script and guide respectively); one observation per arm does not establish a latency improvement. In this calibration the baseline adapter reported an unavailable helper lookup as a completed tool call without returning evidence; do not use this run to compare tool-call success. The adapter now reports the missing helper as unavailable via an error.
