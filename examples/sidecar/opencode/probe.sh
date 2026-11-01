#!/bin/sh
# Non-interactive, model-free probe for the mounted payload. Also run with
# networking disabled to verify that startup does not require a download.
set -eu

tool_dir=/opt/opencode
probe_dir=$(mktemp -d /tmp/opencode-probe.XXXXXX)
trap 'rm -rf "$probe_dir"' EXIT
trap 'exit 1' HUP INT TERM

export XDG_CONFIG_HOME="$probe_dir/config"
export XDG_DATA_HOME="$probe_dir/data"
export XDG_CACHE_HOME="$probe_dir/cache"
export XDG_STATE_HOME="$probe_dir/state"
unset OPENCODE_CONFIG OPENCODE_CONFIG_DIR
export OPENCODE_DISABLE_AUTOUPDATE=1
export OPENCODE_DISABLE_MODELS_FETCH=1
export OPENCODE_DISABLE_PROJECT_CONFIG=1
export OPENCODE_CONFIG_CONTENT='{"autoupdate":false,"plugin":[],"lsp":false,"formatter":false}'

cd "$probe_dir"
version=$("$tool_dir/bin/opencode" --version)
test "$version" = "$(cat "$tool_dir/OPENCODE_VERSION")"
rg_version=$("$tool_dir/bin/rg" --version)
case "$rg_version" in
    "ripgrep $(cat "$tool_dir/RG_VERSION")"*) ;;
    *) echo "Unexpected ripgrep version: $rg_version" >&2; exit 1 ;;
esac
printf '%s\n' sidecar-probe | "$tool_dir/bin/rg" --quiet '^sidecar-probe$'

# Exercise application initialization, config parsing and writable state, beyond
# the early --version exit. No provider credentials or model request are needed.
"$tool_dir/bin/opencode" debug config > "$probe_dir/config.json"
test -s "$probe_dir/config.json"
printf 'OpenCode %s: mounted runtime, ripgrep and config startup OK\n' "$version"
