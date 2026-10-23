ARG BASE_IMAGE=busybox:1.36.1-musl
FROM ${BASE_IMAGE} AS builder
ARG TOOL_VERSION=1.36.1
RUN busybox 2>&1 | grep -F "BusyBox v${TOOL_VERSION} "
FROM scratch
COPY --from=builder /bin/busybox /bin/busybox
