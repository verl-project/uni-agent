# Test Case Submission Guide

## 1. Basic Principles for Test Cases

- Currently, UT test cases that run on the CPU, GPU and NPU backend are supported.
- Submit all test cases to the `tests/uni_agent` directory. Test cases outside this directory will not be run in the gate.
- Under the `tests/uni_agent` directory, submit test cases by functional module.

## 2. Test Case Submission Steps

### 1. Place runtime dependencies uniformly in `requirements-test.txt` in the root directory

### 2. Submit test cases to the designated location according to the principles described above

### 3. Add tags to test cases to specify the runtime backend and test case level

#### Supported tags:

- cpu: Specifies that the test case runs on the CPU backend
- gpu: Specifies that the test case runs on the GPU backend
- npu: Specifies that the test case runs on the NPU backend
- level0: Specifies the test case as a gate-level test case, which will run each time the gate is triggered
- level1: Specifies the test case as a version-level test case, which is only run manually during version verification.

#### Notes:

- Configure tags at the function granularity. Do not configure tags directly on a class.
- Both a backend tag and a test case level tag must be specified for the test case to run in the gate.
- Only one of `level0` and `level1` can be specified for a single test case.
- To trigger GPU/NPU backend test cases, configure the corresponding modified file paths in the trigger rules; see Section 4.

#### Test case example

Script name format: `test_xxx.py`

```

import pytest


@pytest.mark.cpu
@pytest.mark.level0
def test_xxxx():
    ...

```

### 4. Submit the datasets and weights required by the test cases to the gate machine

Not supported for now; to be added.

## 3. Gate Test Case Verification Method

### 1. Logic verification

In the "checks" tab of the PR, find the `tests-cpu`, `tests-gpu` and `tests-npu` workflow; click to view the corresponding gate logs.

The logs contain `test case runtime information` and `coverage statistics results`.

Ensure that the test cases run without errors before merging.

### 2. Coverage verification

**Currently, this verification is only a recommendation and is not set as a mandatory check**

After CI workflow finishes running, a coverage comment will be added to the PR, showing the coverage of the code corresponding to the changes in this PR.

This includes: `coverage of the modified code` and `coverage details displayed by code line`.

It is recommended to improve the test case coverage based on the results.

## 4. Configure Precise GPU/NPU Trigger Rules

### 1. Mechanism Description

This mechanism establishes a mapping between project files and backend test suites so that after a file is modified, only the tests for the associated backend are triggered.

Triggering principles:

- For files configured with a GPU/NPU backend mapping, changes trigger the corresponding backend test cases.
- For files without a configured backend mapping, changes will trigger CPU backend test cases.

### 2. Configuration Method

Modify the configuration file: [filter-paths.yml](uni_agent/filter-paths.yml)

Configure file paths under the corresponding backend:

```

gpu:
  - '<Changes under this path will trigger GPU backend tests>'
npu:
  - '<Changes under this path will trigger NPU backend tests>'
cpu:
  - '!<Changes under this path will not trigger CPU backend tests>'

```

- gpu/npu: Paths that trigger the corresponding backend tests.
- cpu: Paths whose changes will not trigger CPU backend tests; use `!` to mark exclusion paths. Paths not configured default to triggering CPU backend tests.
