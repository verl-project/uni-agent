### What does this PR do?

> Add a concise, reviewer-friendly overview of what this PR changes and why. Reference related GitHub issues or prior PRs if they help with review.

### Checklist Before Starting

- [ ] Search for similar PRs or issues and paste at least one relevant link here: ...
- [ ] Format the PR title as `[{modules}] {type}: {description}` (checked by CI)
  - `{modules}` may include `core`, `interaction`, `model`, `env`, `tools`, `deployment`, `reward`, `dashboard`, `docs`, `examples`, `data`, `train`, `ci`, `build`, `deps`, `misc`
  - If this PR involves multiple modules, separate them with `,` like `[interaction, tools, docs]`
  - `{type}` must be one of `feat`, `fix`, `refactor`, `chore`, `test`
  - If this PR breaks an API, config contract, workflow, or other compatibility boundary, add `[BREAKING]` to the beginning of the title
  - For a stacked PR series, you may prepend a progress marker such as `[1/N]`
  - Example: `[BREAKING][deployment, docs] feat: simplify runtime env configuration`

### Test

> List the checks you ran. If CI coverage is not practical for this change, describe the manual validation or experiment results.

### API and Usage Example

> Show any public interface changes or updated usage examples if relevant.

```python
# Add a short example here when the PR changes public behavior
```

### Design & Code Changes

> Summarize the approach for non-trivial changes and call out important implementation details or trade-offs.

### Checklist Before Submitting

- [ ] Read the [Contribute Guide](../CONTRIBUTING.md)
- [ ] Run `pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always`
- [ ] Add or update docs/examples for user-facing changes
- [ ] Add tests or explain why tests are not practical
- [ ] Confirm the PR title matches the required format
- [ ] Confirm the placeholder text in this template has been replaced with real content
