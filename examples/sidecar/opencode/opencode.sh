#!/bin/sh
set -eu

# Prefer caller-supplied or task-system trust; use the bundled public roots when
# the task image has no certificate bundle. No shared-library environment leaks
# into the task's shell, Python or other subprocesses.
if [ -z "${SSL_CERT_FILE:-}" ]; then
    if [ -r /etc/ssl/certs/ca-certificates.crt ]; then
        SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    elif [ -r /etc/pki/tls/certs/ca-bundle.crt ]; then
        SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
    else
        SSL_CERT_FILE=/opt/opencode/share/ca-certificates.crt
    fi
    export SSL_CERT_FILE
fi

PATH="/opt/opencode/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export PATH
# OpenCode v2 self-spawns via process.execPath, which is the explicit loader
# below. Supervise its private server explicitly instead; leave 1.x unchanged.
if [ "$(cat /opt/opencode/OPENCODE_VERSION)" = 2.2.5 ]; then
    for arg in "$@"; do
        [ "$arg" != -- ] || break
        if [ "$arg" = --standalone ]; then
            exec /usr/bin/python3 /opt/opencode/libexec/standalone.py "$@"
        fi
    done
fi
# Override musl's lookup for this executable only. Keep the caller's environment
# intact for task tools: clearing or exporting a replacement LD_LIBRARY_PATH
# would break tools that need their own glibc/custom libraries.
exec /opt/opencode/lib/ld.so --library-path /opt/opencode/lib /opt/opencode/libexec/opencode "$@"
