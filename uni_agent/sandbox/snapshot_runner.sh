# Sandbox-side snapshot runner: needs only bash and coreutils (sleep, head, wc, mv, rm).
# usage: bash runner.sh ROOT ID TIMEOUT LIMIT STRICT [COMMAND]
#   COMMAND omitted: ROOT/cmd.ID was uploaded beforehand.
#   STRICT=1: a failed state restore (e.g. deleted cwd) skips the command;
#   STRICT=0: fall back to $HOME (or /) and run it.
#   ROOT/profile, if present, holds static functions/aliases sourced before the state.
# User stdout -> our stdout, user stderr -> our stderr, then one trailer line
#   "UASH1 <rc> <committed> <timed_out> <ran> <stdout_bytes> <stderr_bytes>"
# on stderr. It is always the last line written, so user output cannot forge it.
# The hot path uses builtins only: every avoided fork is ~1 ms inside a container.
set -u
root=$1 id=$2 limit=$3 cap=$4 strict=$5
state=$root/state cmd=$root/cmd.$id pending=$root/pending.$id done=$root/done.$id
ran=$root/ran.$id out=$root/out.$id err=$root/err.$id flag=$root/timeout.$id
[ $# -ge 6 ] && printf %s "$6" >"$cmd"

printf -v q_pending %q "$pending"
printf -v q_done %q "$done"
printf -v q_state %q "$state"
printf -v q_cmd %q "$cmd"
printf -v q_ran %q "$ran"
save="set +e +u; { builtin export -p; builtin printf 'cd -- %q || return\\n' \"\$PWD\"; } >$q_pending && builtin printf saved >$q_done"
printf -v q_trap %q "builtin trap - EXIT; $save"
if [ "$strict" = 1 ]; then
    restore="set -e; builtin source $q_state; set +e"
else
    restore="builtin source $q_state || builtin cd -- \"\${HOME:-/}\" 2>/dev/null || builtin cd /"
fi
if [ -e "$root/profile" ]; then
    printf -v q_profile %q "$root/profile"
    restore="builtin source $q_profile; $restore"
fi
# Save after a normal return too, so a user EXIT trap does not drop the snapshot;
# the EXIT trap covers exit and errexit. The status is parked in \$1 because
# positional parameters are not part of the snapshot, unlike any variable name.
script="$restore
: >$q_ran
builtin trap $q_trap EXIT
builtin eval \"\$(<$q_cmd)\"
builtin set -- \"\$?\"; $save; builtin exit \"\$1\""

# Job control gives the command and the watchdog their own process groups.
# Output goes to files: a background job holding stdout cannot delay completion.
set -m
env -i "$BASH" --noprofile --norc -c "$script" </dev/null >"$out" 2>"$err" &
pid=$!
{ sleep "$limit"; : >"$flag"; kill -KILL -- "-$pid"; } >/dev/null 2>&1 &
watchdog=$!
disown "$watchdog"
set +m
wait "$pid"
rc=$?
kill -KILL -- "-$watchdog" >/dev/null 2>&1
timed=0 committed=0 started=0 osz=0 esz=0
[ -e "$flag" ] && timed=1
[ -e "$ran" ] && started=1
if [ "$timed" = 0 ] && [ -e "$done" ] && [ -e "$pending" ]; then
    mv -f "$pending" "$state" && committed=1
fi
if [ -s "$out" ]; then osz=$(wc -c <"$out"); head -c "$cap" "$out"; fi
if [ -s "$err" ]; then esz=$(wc -c <"$err"); head -c "$cap" "$err" >&2; fi
printf '\nUASH1 %d %d %d %d %d %d\n' "$rc" "$committed" "$timed" "$started" "$osz" "$esz" >&2
rm -f "$cmd" "$pending" "$done" "$ran" "$out" "$err" "$flag"
