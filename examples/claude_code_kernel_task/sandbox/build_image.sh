#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to your prepared Ascend/Claude Code image}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-triton-claude-code-env:latest}
SANDBOX_USER=${SANDBOX_USER:-claude}

# Keep upstream assets out of the recipe; materialize the exact revision at build time.
CANNBOT_COMMIT=49451c99df99d0dc31f2a8acb920e46e1ee52105
BUILD_DIR=$(mktemp -d)
trap 'rm -rf -- "${BUILD_DIR}"' EXIT
mkdir -p "${BUILD_DIR}/upstream" "${BUILD_DIR}/context"
if [[ -n "${CANNBOT_SOURCE:-}" ]]; then
  source_repo=${CANNBOT_SOURCE}
else
  source_repo="${BUILD_DIR}/source"
  git init -q "${source_repo}"
  git -C "${source_repo}" fetch --quiet --depth=1 \
    https://gitcode.com/cann/cannbot-skills.git "${CANNBOT_COMMIT}"
fi
git -C "${source_repo}" archive "${CANNBOT_COMMIT}" \
  ops/npu-arch/references/npu-arch-guide-triton.md \
  ops/triton-latency-optimizer ops/triton-op-coding ops/triton-op-designer \
  ops/triton-op-verifier | tar -x -C "${BUILD_DIR}/upstream"

context="${BUILD_DIR}/context"
cp "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}/.dockerignore" "${context}/"
cp -R "${SCRIPT_DIR}/template" "${SCRIPT_DIR}/tools" "${context}/"
skills="${context}/template/.claude/skills"
mkdir -p "${skills}/npu-arch/references" "${skills}/triton-op-verifier" "${context}/tools/verifier"
for skill in triton-latency-optimizer triton-op-coding triton-op-designer; do
  cp -R "${BUILD_DIR}/upstream/ops/${skill}" "${skills}/"
done
cp "${BUILD_DIR}/upstream/ops/npu-arch/references/npu-arch-guide-triton.md" "${skills}/npu-arch/references/"
cp "${BUILD_DIR}/upstream/ops/triton-op-verifier/SKILL.md" "${skills}/triton-op-verifier/"
cp "${BUILD_DIR}/upstream/ops/triton-op-verifier/scripts/"*.py "${context}/tools/verifier/"
(cd "${context}" && git -c core.autocrlf=false apply --whitespace=nowarn "${SCRIPT_DIR}/rl-adaptations.patch")

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "SANDBOX_USER=${SANDBOX_USER}" \
  --tag "${OUTPUT_IMAGE}" \
  "${context}"
