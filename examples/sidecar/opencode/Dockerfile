# syntax=docker/dockerfile:1
# Filesystem payload for Linux amd64 task sandboxes, mounted at /opt/opencode.
ARG BASE_IMAGE=alpine:3.22.2@sha256:4b7ce07002c69e8f3d704a9c5d6fd3053be500b7f1c69fc0d80990c2ad8dd412
FROM ${BASE_IMAGE} AS builder

RUN test "$(uname -m)" = x86_64 \
    && apk add --no-cache ca-certificates curl libstdc++ python3

ARG OPENCODE_VERSION=1.18.18
ARG OPENCODE_PACKAGE=opencode-linux-x64-baseline-musl
ARG NPM_REGISTRY=https://registry.npmjs.org
ARG OPENCODE_SHA512=7170f580067e4d399d137b675b2e6a3ecb598eb9db3e0d2b3d4c2d70778b69e07e298c50584a227b573cf2862665f43e45623afbff8b2c5d05536f54c9dd4daf
ARG RG_VERSION=15.1.0
ARG RG_SHA256=1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599

# The platform npm package contains the native executable; Node/npm are not
# runtime dependencies. Baseline avoids AVX2. The musl build needs a dynamic loader
# and C++ libraries; copy these into the payload rather than relying on the task OS.
RUN mkdir -p /opt/opencode/bin /opt/opencode/libexec /opt/opencode/lib /opt/opencode/etc /opt/opencode/share \
    && curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 \
      "${NPM_REGISTRY%/}/${OPENCODE_PACKAGE}/-/${OPENCODE_PACKAGE##*/}-${OPENCODE_VERSION}.tgz" \
      -o /tmp/opencode.tar.gz \
    && echo "${OPENCODE_SHA512}  /tmp/opencode.tar.gz" | sha512sum -c - \
    && tar -xzf /tmp/opencode.tar.gz -C /opt/opencode/libexec --strip-components=2 package/bin/opencode \
    && cp -L /lib/ld-musl-x86_64.so.1 /opt/opencode/lib/ld.so \
    && cp -L /usr/lib/libstdc++.so.6 /usr/lib/libgcc_s.so.1 /opt/opencode/lib/ \
    && ln -s ld.so /opt/opencode/lib/libc.musl-x86_64.so.1 \
    && printf '%s\n' /opt/opencode/lib > /opt/opencode/etc/ld-musl-x86_64.path \
    && cp /etc/ssl/certs/ca-certificates.crt /opt/opencode/share/ca-certificates.crt \
    && printf '%s\n' "${OPENCODE_VERSION}" > /opt/opencode/OPENCODE_VERSION \
    && rm /tmp/opencode.tar.gz

# Preserve Bun's embedded data offsets: replacing PT_INTERP in place avoids the
# executable corruption caused by ELF relayout. musl reads the adjacent etc/
# search-path file, so this does not export LD_LIBRARY_PATH to task subprocesses.
RUN python3 - <<'PY'
import struct

with open('/opt/opencode/libexec/opencode', 'r+b') as binary:
    header = binary.read(64)
    assert header[:6] == b'\x7fELF\x02\x01', 'Expected a little-endian ELF64 executable'
    table = struct.unpack_from('<Q', header, 32)[0]
    size, count = struct.unpack_from('<HH', header, 54)
    for index in range(count):
        binary.seek(table + index * size)
        kind, _, offset, _, _, length, _, _ = struct.unpack('<IIQQQQQQ', binary.read(56))
        if kind != 3:
            continue
        binary.seek(offset)
        assert binary.read(length) == b'/lib/ld-musl-x86_64.so.1\0', 'Unexpected ELF interpreter'
        replacement = b'/opt/opencode/lib/ld.so\0'
        assert len(replacement) <= length, 'Interpreter path does not fit'
        binary.seek(offset)
        binary.write(replacement.ljust(length, b'\0'))
        break
    else:
        raise RuntimeError('ELF interpreter not found')
PY

# OpenCode otherwise downloads ripgrep on demand when it is absent from PATH.
RUN curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 \
      "https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
      -o /tmp/rg.tar.gz \
    && echo "${RG_SHA256}  /tmp/rg.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/rg.tar.gz -C /tmp \
    && cp "/tmp/ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl/rg" /opt/opencode/bin/rg \
    && printf '%s\n' "${RG_VERSION}" > /opt/opencode/RG_VERSION \
    && rm -rf /tmp/rg.tar.gz "/tmp/ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl"

COPY opencode.sh /opt/opencode/bin/opencode
COPY probe.sh /opt/opencode/bin/probe.sh
RUN chmod 755 /opt/opencode/bin/opencode /opt/opencode/bin/probe.sh \
    && /opt/opencode/bin/probe.sh

# Mount the image root at /opt/opencode, including the relocated ELF interpreter.
# The task image supplies /bin/sh and its project/test environment. Writable
# OpenCode configuration, cache and state belong outside this read-only mount.
FROM scratch AS tool
COPY --from=builder /opt/opencode/ /
