# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CUR_DIR := $(shell pwd)
LOG_DIR := $(abspath $(CUR_DIR)/logs/)
TEST_DIR := $(abspath $(CUR_DIR)/../)
REPORT_DIR ?= $(CUR_DIR)/report
CAPE_TEST_IMAGE ?= docker://nvcr.io/nvidia/pytorch:26.06-py3
CAPE_URL ?= http://127.0.0.1:18892
CAPE_USER ?= user-000
RUN_ID ?= manual
BUNDLE ?= $(abspath $(CUR_DIR)/state)
CAPE_NODE_BUNDLE ?= $(abspath $(CUR_DIR)/evidence)
CAPE_STATE_DIR ?= $(BUNDLE)/controller
CAPE_ADMIN_TOKEN_FILE ?= $(BUNDLE)/credentials/admin.json
CAPE_USER_TOKEN_FILE ?= $(BUNDLE)/credentials/$(CAPE_USER).json
CAPE_SHARED_GPU_SLOTS ?= 16
export CAPE_URL CAPE_USER RUN_ID BUNDLE CAPE_STATE_DIR CAPE_ADMIN_TOKEN_FILE
export CAPE_USER_TOKEN_FILE CAPE_SHARED_GPU_SLOTS
export CAPE_NODE_BUNDLE
# Optional: a calibrated processing constant (median of a few direct
# calibrated_workload.sh runs in the same execution environment). Marker
# durations are always measured; the constant only enables the no-marker
# queuing fallback and the processing-vs-calibration anomaly cross-check.
PROCESSING_CONSTANT ?=
_PROCESSING_CONSTANT_ARGS := $(if $(PROCESSING_CONSTANT),--processing-constant $(PROCESSING_CONSTANT))

.DELETE_ON_ERROR:
.SILENT:

all: concurrency

concurrency: validate_processing_constant exclusive_host

validate_processing_constant:
	@if [ -n "$(PROCESSING_CONSTANT)" ]; then \
		case "$(PROCESSING_CONSTANT)" in \
			*[!0-9.]*|*.*.*|'') echo 'PROCESSING_CONSTANT must be a non-negative number' >&2; exit 2 ;; \
		esac; \
		case "$(PROCESSING_CONSTANT)" in \
			*[0-9]*) ;; \
			*) echo 'PROCESSING_CONSTANT must contain a digit' >&2; exit 2 ;; \
		esac; \
	fi

prerequisites:
	bash prerequisites.sh

controller:
	mkdir -p $(LOG_DIR)
	bash launch_role.sh --pid-file $(LOG_DIR)/controller.pid \
		--log-file $(LOG_DIR)/controller.out --label controller -- \
		env CAPE_REPLACE_EXISTING=1 bash controller.sh

user:
	bash user.sh

register_exclusive:
	CAPE_POOL=exclusive-pool CAPE_POOL_MODE=exclusive CAPE_GPU_SLOTS=1 CAPE_GPU_COUNT=1 \
	bash launch_role.sh --pid-file $(LOG_DIR)/exclusive_node.pid \
		--log-file $(LOG_DIR)/exclusive_node.out --label exclusive-node -- bash register_node.sh --controller-url "$(CAPE_URL)" --token-file "$(CAPE_USER_TOKEN_FILE)"

register_shared:
	CAPE_POOL=shared-pool CAPE_POOL_MODE=shared CAPE_GPU_SLOTS=$(CAPE_SHARED_GPU_SLOTS) CAPE_GPU_COUNT=1 \
	bash launch_role.sh --pid-file $(LOG_DIR)/shared_node.pid \
		--log-file $(LOG_DIR)/shared_node.out --label shared-node -- bash register_node.sh --controller-url "$(CAPE_URL)" --token-file "$(CAPE_USER_TOKEN_FILE)"

exclusive_util:
	bash status.sh --token-file "$(CAPE_USER_TOKEN_FILE)" --controller-url "$(CAPE_URL)" --pool exclusive-pool utilization

shared_util:
	bash status.sh --token-file "$(CAPE_USER_TOKEN_FILE)" --controller-url "$(CAPE_URL)" --pool shared-pool utilization

exclusive_ready:
	bash wait_pool_ready.sh --token-file "$(CAPE_USER_TOKEN_FILE)" exclusive-pool

shared_ready:
	bash wait_pool_ready.sh --token-file "$(CAPE_USER_TOKEN_FILE)" shared-pool

# Workload for both host and mix targets; command-line WORKLOAD_ARGS overrides this.
# Default: CPU sleep 1s followed by fixed-work GPU vector addition.
WORKLOAD_ARGS ?= --cpu-seconds 1 --gpu-mode vecadd
# To use CPU sleep 1s + one GPU CTA sleeping 1.5s, replace the line above with:
# WORKLOAD_ARGS ?= --cpu-seconds 1 --gpu-mode sleep --gpu-single-cta --gpu-seconds 1.5
_WORKLOAD_ARGS := $(WORKLOAD_ARGS)

# 1:1 mix of host-stateless and image-stateless requests
_IMAGE_CACHE_DIR := $(abspath $(REPORT_DIR)/mix/image/)
_HOST_CACHE_DIR := $(abspath $(REPORT_DIR)/mix/host/)
exclusive_mix: exclusive_ready
	@mkdir -p $(_IMAGE_CACHE_DIR) $(_HOST_CACHE_DIR)
	bash concurrency.sh --token-file "$(CAPE_USER_TOKEN_FILE)" \
		--workloads 16 \
		--concurrency 0 \
		--arrival-schedule poisson:2 \
		--seed 42 \
		--class 'host:1:--isolation-policy host-stateless --bind $(_HOST_CACHE_DIR)/:/tmp/cache/:rw' \
		--class 'image:1:--isolation-policy image-stateless --image $(CAPE_TEST_IMAGE) --bind $(_IMAGE_CACHE_DIR):/tmp/cache/:rw' \
		$(_PROCESSING_CONSTANT_ARGS) \
		--report $(REPORT_DIR)/mix/report.txt \
		-- \
		--controller-url "$(CAPE_URL)" --pool exclusive-pool \
		--user "$(CAPE_USER)" \
		--gpus 1 \
		--bind $(TEST_DIR)/:/workspace:rw \
		--cwd /workspace/ --workspace-target /cape-workspace \
		-- bash -c 'PATH="$$PATH:/usr/local/cuda/bin" exec bash /workspace/calibrated_workload.sh $(_WORKLOAD_ARGS) --cache-dir /tmp/cache/gpu-sleep-cache'
shared_mix: shared_ready
	@mkdir -p $(_IMAGE_CACHE_DIR) $(_HOST_CACHE_DIR)
	bash concurrency.sh --token-file "$(CAPE_USER_TOKEN_FILE)" \
		--workloads 16 \
		--concurrency 0 \
		--arrival-schedule poisson:2 \
		--seed 42 \
		--class 'host:1:--isolation-policy host-stateless --bind $(_HOST_CACHE_DIR)/:/tmp/cache/:rw' \
		--class 'image:1:--isolation-policy image-stateless --image $(CAPE_TEST_IMAGE) --bind $(_IMAGE_CACHE_DIR):/tmp/cache/:rw' \
		$(_PROCESSING_CONSTANT_ARGS) \
		--report $(REPORT_DIR)/mix/report.txt \
		-- \
		--controller-url "$(CAPE_URL)" --pool shared-pool \
		--user "$(CAPE_USER)" \
		--gpus 1 \
		--bind $(TEST_DIR)/:/workspace:rw \
		--cwd /workspace/ --workspace-target /cape-workspace \
		-- bash -c 'PATH="$$PATH:/usr/local/cuda/bin" exec bash /workspace/calibrated_workload.sh $(_WORKLOAD_ARGS) --cache-dir /tmp/cache/gpu-sleep-cache'

# full host-stateless requests
_HOST_CACHE_DIR := $(abspath $(REPORT_DIR)/host/)
# Provision this same node-local directory for every participating GPU command.
GPU_LOCK_DIR ?= /tmp/cape-gpulock-$(shell id -u)
exclusive_host: exclusive_ready
	@mkdir -p $(_HOST_CACHE_DIR)
	bash concurrency.sh --token-file "$(CAPE_USER_TOKEN_FILE)" \
		--workloads 16 \
		--concurrency 0 \
		--arrival-schedule poisson:2 \
		--seed 42 \
		--class 'host:1:--isolation-policy host-stateless --bind $(_HOST_CACHE_DIR)/:/tmp/cache/:rw' \
		$(_PROCESSING_CONSTANT_ARGS) \
		--report $(REPORT_DIR)/host/exclusive_host_report.txt \
		-- \
		--controller-url "$(CAPE_URL)" --pool exclusive-pool \
		--user "$(CAPE_USER)" \
		--gpus 1 \
		--bind $(TEST_DIR)/:/workspace:rw \
		--cwd /workspace/ --workspace-target /cape-workspace \
		-- bash -c 'PATH="$$PATH:/usr/local/cuda/bin" exec bash /workspace/calibrated_workload.sh $(_WORKLOAD_ARGS) --cache-dir /tmp/cache/gpu-sleep-cache'

	python3 gpu_timing_report.py --report "$(REPORT_DIR)/host/exclusive_host_report.txt" --cache-dir "$(_HOST_CACHE_DIR)/gpu-sleep-cache"

shared_host shared_host_file_lock: shared_ready
	@mkdir -p $(_HOST_CACHE_DIR)
	$(if $(filter shared_host_file_lock,$@),mkdir -p "$(GPU_LOCK_DIR)",:)
	bash concurrency.sh --token-file "$(CAPE_USER_TOKEN_FILE)" \
		--workloads 16 \
		--concurrency 0 \
		--arrival-schedule poisson:2 \
		--seed 42 \
		--class 'host:1:--isolation-policy host-stateless --bind $(_HOST_CACHE_DIR)/:/tmp/cache/:rw' \
		$(_PROCESSING_CONSTANT_ARGS) \
		--report $(REPORT_DIR)/host/$@_report.txt \
		-- \
		--controller-url "$(CAPE_URL)" --pool shared-pool \
		--user "$(CAPE_USER)" \
		--gpus 1 \
		--bind $(TEST_DIR)/:/workspace:rw \
		$(if $(filter shared_host_file_lock,$@),--bind "$(GPU_LOCK_DIR):/runtime/gpulock:rw",) \
		--cwd /workspace/ --workspace-target /cape-workspace \
		-- bash -c 'PATH="$$PATH:/usr/local/cuda/bin" exec bash /workspace/calibrated_workload.sh $(_WORKLOAD_ARGS) $(if $(filter shared_host_file_lock,$@),--gpu-lock-dir /runtime/gpulock,) --cache-dir /tmp/cache/gpu-sleep-cache'
	python3 gpu_timing_report.py --report "$(REPORT_DIR)/host/$@_report.txt" --cache-dir "$(_HOST_CACHE_DIR)/gpu-sleep-cache"

CAPE_SHUTDOWN_TIMEOUT ?= 120
CAPE_SHUTDOWN_FORCE ?=
CAPE_SHUTDOWN_SETTLE ?= 15

shutdown:
	CAPE_SHUTDOWN_TIMEOUT=$(CAPE_SHUTDOWN_TIMEOUT) \
	CAPE_SHUTDOWN_SETTLE=$(CAPE_SHUTDOWN_SETTLE) \
	CAPE_SHUTDOWN_FORCE=$(if $(CAPE_SHUTDOWN_FORCE),1,0) \
	bash shutdown.sh --log-dir $(LOG_DIR)

.PHONY: user all concurrency validate_processing_constant prerequisites controller register_exclusive register_shared exclusive_util shared_util exclusive_ready shared_ready exclusive_mix shared_mix exclusive_host shared_host shared_host_file_lock shutdown clean

clean: shutdown
	python3 clean.py --root "$(CUR_DIR)" --logs "$(LOG_DIR)" \
		--reports "$(REPORT_DIR)" --bundle "$(BUNDLE)" --state "$(CAPE_STATE_DIR)"
	
