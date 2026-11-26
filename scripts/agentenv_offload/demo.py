#!/usr/bin/env python3
"""Run a single-node AgentEnv resident versus pause/resume experiment.

Lifecycle uses HTTP; commands use the pinned aenv CLI on the node over SSH.
Credentials are read from a private TOML file and never put in argv or logs.
No request is sent to a paused guest until explicit connect has completed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import shlex
import stat
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import tomllib

AGENTENV_SHA = "a27b2db909b035f58696ba8798ea0e285c7bfa17"
HERE = Path(__file__).resolve().parent
ORACLE_FIELDS = (
    "nonce",
    "memory_sha256",
    "memory_bytes",
    "pid",
    "process_started_ns",
    "process_start_ticks",
    "counter",
    "fd_offset",
    "fd_sha256",
    "fd_size",
    "journal_sha256",
    "journal_size",
    "owned_dir",
)
SOURCE_FILES = ("demo.py", "guest_probe.py", "host_sampler.py", "analyze.py")


class ExperimentError(RuntimeError):
    pass


class OracleMismatch(ExperimentError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_credentials(path: Path) -> dict:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ExperimentError("credentials must be a regular file")
    if os.name == "posix" and (stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid()):
        raise ExperimentError("credentials must be owned by the current user with mode 0600")
    with path.open("rb") as stream:
        result = tomllib.load(stream)
    if not isinstance(result.get("api_key"), str) or not result["api_key"]:
        raise ExperimentError("credentials require api_key")
    return result


def archive_sources(output: Path, source_dir: Path = HERE) -> dict[str, str]:
    """Freeze only the named experiment sources, never adjacent configuration."""
    target = output / "source"
    target.mkdir(mode=0o700)
    hashes = {}
    for name in SOURCE_FILES:
        path = source_dir / name
        if not path.is_file():
            if name == "analyze.py":
                continue
            raise ExperimentError(f"required experiment source is missing: {name}")
        data = path.read_bytes()
        with (target / name).open("xb") as stream:
            stream.write(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def clean(value, secrets=()):
    if isinstance(value, dict):
        return {
            key: (
                "[redacted]"
                if any(word in key.lower() for word in ("token", "api_key", "api-key"))
                else clean(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [clean(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


class API:
    def __init__(self, url: str, key: str, timeout: float = 180):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query:
            raise ExperimentError("invalid API URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ExperimentError("plaintext API URL must use a local secure-tunnel endpoint")
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.secrets = [key]
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, method: str, path: str, body=None, *, headers=None, expected=(200,)):
        request_headers = {"Content-Type": "application/json"}
        if headers is None:
            request_headers["X-API-Key"] = self.key
        else:
            request_headers.update(headers)
        request = Request(
            self.url + path,
            method=method,
            headers=request_headers,
            data=None if body is None else json.dumps(body).encode(),
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                status, raw = response.status, response.read()
        except HTTPError as response:
            status, raw = response.code, response.read()
        except (OSError, URLError, TimeoutError) as exc:
            raise ExperimentError(f"HTTP {method} transport outcome unknown: {type(exc).__name__}") from None
        if status not in expected:
            # Error bodies can contain credentials or request echoes.
            raise ExperimentError(f"HTTP {method} {path} returned {status}")
        if not raw:
            return status, {}
        try:
            data = json.loads(raw)
        except ValueError:
            raise ExperimentError(f"HTTP {method} {path} returned invalid JSON") from None
        if isinstance(data, dict):
            for name in ("envdAccessToken", "trafficAccessToken"):
                if isinstance(data.get(name), str):
                    self.secrets.append(data[name])
        return status, data


# The helper receives all arguments as JSON on stdin. No shell interpolation of
# file contents, credentials, guest commands, or process identities is needed.
NODE_HELPER = r"""
import base64,json,os,signal,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin)
def identity(pid,label='sandbox'):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z': return None
        return {'pid':int(pid),'starttime_ticks':int(fields[19]),'label':label,'include_descendants':False}
    except (OSError,ValueError,IndexError): return None
def alive(item):
    found=identity(item['pid'])
    return found is not None and found['starttime_ticks']==item['starttime_ticks']
def atomic(path,value):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value)); os.chmod(temp,0o600); os.replace(temp,path)
action=p['action']
if action=='prepare':
    root=Path(p['root']); root.mkdir(parents=True,exist_ok=True)
    folder=root/p['trial']; folder.mkdir(mode=0o700)
    sampler=folder/'host_sampler.py'; sampler.write_bytes(base64.b64decode(p['sampler']))
    roots=[]
    if p.get('provider_pid_file'):
        roots=json.loads(Path(p['provider_pid_file']).read_text())['processes']
        if any(not alive(item) for item in roots): raise RuntimeError('provider PID registry is stale')
    for pid in p.get('provider_pids',[]):
        item=identity(pid,'provider')
        if item is None: raise RuntimeError('provider PID does not exist')
        roots.append(item)
    if not roots: raise RuntimeError('provider/ublk PID registry is required')
    # A newly launched VM or conversion helper exists before create/connect
    # returns its identity. Track service descendants immediately; an explicit
    # sandbox root can later override its label without double-counting its PID.
    roots=[dict(item,include_descendants=True) for item in roots]
    atomic(folder/'base-pids.json',{'processes':roots})
    atomic(folder/'pids.json',{'run_id':p['trial'],'processes':roots})
    log=(folder/'sampler.log').open('wb')
    child=subprocess.Popen([sys.executable,str(sampler),'--output',str(folder/'metrics.jsonl'),
        '--pid-file',str(folder/'pids.json'),'--interval',str(p['interval'])],
        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    log.close(); time.sleep(0.2)
    item=identity(child.pid,'sampler')
    if item is None: raise RuntimeError('sampler failed to start')
    sample_path=folder/'metrics.jsonl'; deadline=time.monotonic()+10
    while (not sample_path.exists() or not sample_path.read_text().endswith('\n')) and time.monotonic()<deadline:
        time.sleep(.05)
    if not sample_path.exists() or not sample_path.read_text().endswith('\n'):
        os.kill(child.pid,signal.SIGTERM); raise RuntimeError('sampler produced no complete sample')
    print(json.dumps({'folder':str(folder),'sampler':item,'provider_roots':roots,
                     'uname':list(os.uname())}))
elif action=='clock':
    print(json.dumps({'monotonic_seconds':time.monotonic(),'wall_time_ns':time.time_ns()}))
elif action in ('discover','discover_and_register'):
    target=str(Path(p['serial_dir'])/p['sandbox_id']/'firecracker-stdout.log')
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            exe=Path(os.readlink(entry/'exe')).name
            if 'firecracker' not in exe: continue
            if os.readlink(entry/'fd'/'1')!=target: continue
            item=identity(int(entry.name))
            if item: found.append(item)
        except OSError: continue
    if action=='discover_and_register':
        if len(found)!=1: raise RuntimeError('expected exactly one owned Firecracker before registration')
        folder=Path(p['folder']); roots=json.loads((folder/'base-pids.json').read_text())['processes']
        atomic(folder/'pids.json',{'run_id':p['trial'],'processes':roots+found})
    print(json.dumps({'processes':found}))
elif action=='alive':
    print(json.dumps({'alive':[item for item in p['processes'] if alive(item)]}))
elif action=='audit':
    errors=[]
    for required in (p['state_dir'],p['runtime_dir']):
        if not Path(required).is_dir(): errors.append('configured_directory_missing:'+required)
    def listing(path,pattern='*'):
        root=Path(path)
        if not root.exists(): return []
        try: return sorted(x.name for x in root.glob(pattern))
        except OSError: errors.append('listing_failed:'+str(root)); return None
    def artifact(path):
        root=Path(path); result={'path':str(root),'exists':root.exists(),'logical_bytes':0,'allocated_bytes':0}
        seen=set()
        if root.exists():
            try:
                for directory,dirs,files in os.walk(root,followlinks=False):
                    for name in files:
                        item=Path(directory)/name
                        if item.is_symlink(): continue
                        s=item.stat(); key=(s.st_dev,s.st_ino)
                        if key in seen: continue
                        seen.add(key); result['logical_bytes']+=s.st_size
                        result['allocated_bytes']+=s.st_blocks*512
            except OSError: errors.append('artifact_read_failed:'+str(root))
        return result
    sid=p.get('sandbox_id'); owned=[]
    if sid:
        owned=[artifact(Path(p['state_dir'])/'persisted-sandboxes'/'artifacts'/sid),
               artifact(Path(p['state_dir'])/'firecracker-work'/'managed-snapshots'/sid)]
    namespace_directories={str(Path('/run/netns')),str(Path(p['runtime_dir'])/'netns')}
    namespaces={}
    for directory in sorted(namespace_directories):
        names=listing(directory)
        namespaces[directory]=None if names is None else [str(Path(directory)/name) for name in names]
    namespace_paths=(None if any(items is None for items in namespaces.values())
                     else sorted({item for items in namespaces.values() for item in items}))
    print(json.dumps({'audit_schema_version':2,'ublk_devices':listing('/dev','ublkb*'),
        'netns':namespace_paths,'netns_by_directory':namespaces,
        'runtime_entries':listing(p['runtime_dir']),'owned_artifacts':owned,'errors':errors}))
elif action=='register':
    folder=Path(p['folder']); roots=json.loads((folder/'base-pids.json').read_text())['processes']
    atomic(folder/'pids.json',{'run_id':p['trial'],'processes':roots+p['processes']})
    print('{}')
elif action=='exec':
    r=subprocess.run([p['aenv'],'exec',p['sandbox_id'],'--','python3','-c',p['code']],
        capture_output=True,text=True,timeout=p['timeout'])
    print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
elif action=='finish':
    item=p['sampler']
    if alive(item): os.kill(item['pid'],signal.SIGTERM)
    deadline=time.monotonic()+10
    while alive(item) and time.monotonic()<deadline: time.sleep(.1)
    if alive(item): raise RuntimeError('sampler did not stop')
    folder=Path(p['folder']); files={}
    for name in ['metrics.jsonl','sampler.log','pids.json','base-pids.json']:
        path=folder/name
        if path.is_file(): files[name]=base64.b64encode(path.read_bytes()).decode()
    print(json.dumps({'files':files}))
else: raise RuntimeError('unknown action')
"""


class Node:
    def __init__(self, args):
        self.args = args
        self.source_dir = args.output / "source"
        self.ssh = ["ssh"] + (["-F", str(args.ssh_config)] if args.ssh_config else [])
        self.ssh += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.node_ssh]

    def call(self, action: str, **kwargs):
        payload = {"action": action, **kwargs}
        command = "python3 -c " + shlex.quote(NODE_HELPER)
        try:
            result = subprocess.run(
                self.ssh + [command],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.args.operation_timeout + 30,
            )
        except subprocess.TimeoutExpired:
            raise ExperimentError(f"node {action} timed out; remote outcome unknown") from None
        if result.returncode:
            raise ExperimentError(f"node {action} failed (exit {result.returncode}); output withheld")
        try:
            return json.loads(result.stdout)
        except ValueError:
            raise ExperimentError(f"node {action} returned invalid JSON") from None

    def prepare(self, trial: str):
        context = self.call(
            "prepare",
            root=self.args.node_root,
            trial=trial,
            sampler=base64.b64encode((self.source_dir / "host_sampler.py").read_bytes()).decode(),
            provider_pid_file=self.args.provider_pid_file,
            provider_pids=self.args.provider_pid,
            interval=self.args.sample_interval,
        )
        before = time.monotonic()
        clock = self.call("clock")
        after = time.monotonic()
        context["clock_calibration"] = {
            "node_monotonic_seconds": clock["monotonic_seconds"],
            "client_midpoint_monotonic_seconds": (before + after) / 2,
            "client_minus_node_seconds": (before + after) / 2 - clock["monotonic_seconds"],
            "uncertainty_seconds": (after - before) / 2,
            "node_wall_time_ns": clock["wall_time_ns"],
        }
        return context

    def discover(self, sid: str):
        return self.call("discover", sandbox_id=sid, serial_dir=self.args.serial_dir)["processes"]

    def discover_and_register(self, context: dict, trial: str, sid: str):
        return self.call(
            "discover_and_register",
            sandbox_id=sid,
            serial_dir=self.args.serial_dir,
            folder=context["folder"],
            trial=trial,
        )["processes"]

    def alive(self, processes: list):
        return self.call("alive", processes=processes)["alive"]

    def audit(self, sid=None):
        return self.call("audit", sandbox_id=sid, state_dir=self.args.state_dir, runtime_dir=self.args.runtime_dir)

    def register(self, context: dict, trial: str, processes: list):
        self.call("register", folder=context["folder"], trial=trial, processes=processes)

    def launch_probe(self, sid: str):
        encoded = base64.b64encode((self.source_dir / "guest_probe.py").read_bytes()).decode()
        config = {
            "memory": self.args.memory_mib,
            "entropy": self.args.entropy,
            "seed": self.args.seed,
            "dirty": self.args.dirty_ratio,
        }
        bootstrap = (
            "import base64,json,subprocess,time;from pathlib import Path;"
            "root=Path('/tmp/offload-probe');root.mkdir(exist_ok=True);"
            f"(root/'guest_probe.py').write_bytes(base64.b64decode({encoded!r}));"
            f"c={config!r};"
            "log=(root/'startup.log').open('wb');"
            "p=subprocess.Popen(['python3',str(root/'guest_probe.py'),'--workdir',str(root),"
            "'--memory-mib',str(c['memory']),'--port','8000','--entropy',c['entropy'],"
            "'--seed',str(c['seed']),'--dirty-ratio',str(c['dirty'])],"
            "stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True);"
            "log.close();print(json.dumps({'probe_pid':p.pid}))"
        )
        started = time.monotonic()
        result = self.call(
            "exec", aenv=self.args.aenv_bin, sandbox_id=sid, code=bootstrap, timeout=self.args.operation_timeout
        )
        elapsed = time.monotonic() - started
        if result["returncode"]:
            raise ExperimentError(f"aenv probe bootstrap failed (exit {result['returncode']})")
        return elapsed

    def finish(self, context: dict, output: Path):
        data = self.call("finish", folder=context["folder"], sampler=context["sampler"])
        for name, content in data["files"].items():
            if name not in {"metrics.jsonl", "sampler.log", "pids.json", "base-pids.json"}:
                raise ExperimentError("unexpected node artifact")
            (output / name).write_bytes(base64.b64decode(content))


def assert_continuity(before: dict, after: dict) -> None:
    mismatches = [key for key in ORACLE_FIELDS if key not in before or before.get(key) != after.get(key)]
    if mismatches:
        raise OracleMismatch("runtime oracle mismatch: " + ", ".join(mismatches))


class Trial:
    def __init__(self, args, api: API, node: Node, mode: str, trial_id: str):
        self.args, self.api, self.node, self.mode, self.id = args, api, node, mode, trial_id
        self.output = args.output / trial_id
        self.output.mkdir(mode=0o700)
        self.sid = None
        self.traffic_token = None
        self.lifecycle = "not_created"
        self.busy = False
        self.unknown_command = False
        self.known_processes = []
        self.current_processes = []
        self.context = None
        self.create_attempted = False
        self.measurement_started = None
        self.baseline_audit = None
        self.cleanup = {"confirmed": False}
        self.ttl = max(600, math.ceil(args.cycles * (args.wait_seconds + args.operation_timeout * 3) + 300))

    def event(self, name: str, **data):
        row = {
            "event": name,
            "trial_id": self.id,
            "mode": self.mode,
            "sandbox_id": self.sid,
            "monotonic_seconds": time.monotonic(),
            "wall_time_ns": time.time_ns(),
            **data,
        }
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(clean(row, self.api.secrets), allow_nan=False) + "\n")

    def proxy(self, method: str, path: str, body=None):
        if self.lifecycle != "running":
            raise ExperimentError("guest request forbidden outside confirmed running state")
        headers = {
            "x-agentenv-sandbox-id": self.sid,
            "x-agentenv-target-port": "8000",
            "e2b-traffic-access-token": self.traffic_token,
        }
        return self.api.request(method, "/proxy" + path, body, headers=headers)[1]

    def refresh(self):
        self.api.request(
            "POST", f"/sandboxes/{self.sid}/refreshes", {"duration": min(3600, self.ttl)}, expected=(200, 204)
        )

    def observe_node(self, phase: str):
        try:
            _, data = self.api.request("GET", "/nodes", expected=(200, 404))
            self.event("node_observation", phase=phase, nodes=data)
        except ExperimentError as exc:
            self.event("node_observation_unavailable", phase=phase, error=str(exc))

    def find_process(self):
        started = time.monotonic()
        found = self.node.discover_and_register(self.context, self.id, self.sid)
        if len(found) != 1:
            raise ExperimentError(f"expected exactly one owned Firecracker, found {len(found)}")
        self.current_processes = found
        for item in found:
            if item not in self.known_processes:
                self.known_processes.append(item)
        elapsed = time.monotonic() - started
        self.event(
            "firecracker_identity",
            processes=found,
            sandbox_instance_id="unknown",
            ssh_discovery_registration_seconds=elapsed,
        )
        return elapsed

    def pause(self):
        if self.lifecycle != "running" or self.busy or self.unknown_command:
            raise ExperimentError("pause rejected: active or unknown command, or wrong lifecycle state")
        if not self.current_processes:
            raise ExperimentError("pause rejected: Firecracker identity is unobserved")
        self.lifecycle = "pausing"
        started = time.monotonic()
        self.event("pause_requested")
        self.api.request("POST", f"/sandboxes/{self.sid}/pause", {}, expected=(204,))
        self.event("pause_api_done")
        deadline = time.monotonic() + self.args.operation_timeout
        while self.node.alive(self.current_processes):
            if time.monotonic() >= deadline:
                raise ExperimentError("pause returned but owned Firecracker is still alive")
            time.sleep(0.1)
        self.lifecycle = "paused"
        self.node.register(self.context, self.id, [])
        self.event(
            "resources_released",
            elapsed=time.monotonic() - started,
            evidence="owned Firecracker PID/starttime gone; memory measured separately",
        )
        self.observe_node("paused")
        return time.monotonic() - started

    def resume(self):
        if self.lifecycle != "paused":
            raise ExperimentError("resume rejected outside paused state")
        self.lifecycle = "resuming"
        started = time.monotonic()
        self.event("resume_requested")
        status, data = self.api.request(
            "POST", f"/v2/sandboxes/{self.sid}/connect", {"timeout": self.ttl}, expected=(200, 201)
        )
        if status != 201:
            raise ExperimentError("resume returned an already-running instance; unexpected wakeup")
        if data.get("sandboxID") != self.sid:
            raise ExperimentError("resume returned a different sandbox identity")
        self.traffic_token = data.get("trafficAccessToken", self.traffic_token)
        api_seconds = time.monotonic() - started
        self.event("resume_api_done", elapsed=api_seconds)
        ssh_seconds = self.find_process()
        self.lifecycle = "running"
        self.last_resume_parts = {"resume_api_seconds": api_seconds, "resume_ssh_observation_seconds": ssh_seconds}
        return time.monotonic() - started

    def tool(self, cycle: int):
        if self.busy or self.unknown_command:
            raise ExperimentError("command admission rejected")
        self.busy = True
        self.event("tool_dispatched", command_id=f"{self.id}-{cycle}")
        try:
            result = self.proxy(
                "POST", "/tool", {"command_id": f"{self.id}-{cycle}", "read_bytes": 17, "payload": f"cycle-{cycle}"}
            )
            if not result.get("ok") or result.get("duplicate"):
                raise ExperimentError("tool response is not a new confirmed completion")
            self.event("tool_completed", result=result)
            return result["state"]
        except Exception:
            self.unknown_command = True
            raise
        finally:
            self.busy = False

    def delete(self):
        if self.sid is None:
            self.cleanup = {
                "confirmed": not self.create_attempted,
                "reason": "create outcome unknown" if self.create_attempted else "create not attempted",
            }
            return
        before_delete = self.lifecycle
        self.lifecycle = "deleting"
        self.event("delete_requested", from_state=before_delete)
        self.api.request("DELETE", f"/sandboxes/{self.sid}", expected=(200, 204, 404))
        deadline = time.monotonic() + self.args.operation_timeout
        while True:
            status, _ = self.api.request("GET", f"/sandboxes/{self.sid}", expected=(200, 404))
            alive = self.node.alive(self.known_processes)
            undiscovered = self.node.discover(self.sid)
            if status == 404 and not alive and not undiscovered:
                audit = self.node.audit(self.sid)
                before = self.baseline_audit or {}
                new_devices = sorted(set(audit.get("ublk_devices") or []) - set(before.get("ublk_devices") or []))
                new_netns = sorted(set(audit.get("netns") or []) - set(before.get("netns") or []))
                remaining = [item for item in audit.get("owned_artifacts", []) if item["exists"]]
                observable = bool(
                    self.baseline_audit is not None and not audit.get("errors") and not before.get("errors")
                )
                complete = observable and not new_devices and not new_netns and not remaining
                self.cleanup = {
                    "confirmed": complete,
                    "api_pid_confirmed": True,
                    "api_absent": True,
                    "owned_firecracker_gone": True,
                    "resource_audit": {
                        "before": before,
                        "after": audit,
                        "new_ublk_devices": new_devices,
                        "new_netns": new_netns,
                        "owned_artifacts_remaining": remaining,
                        "status": "no_observed_residue" if complete else "pending",
                    },
                }
                self.lifecycle = "deleted"
                self.event("cleanup_confirmed" if complete else "cleanup_resource_audit_pending", **self.cleanup)
                return
            if time.monotonic() >= deadline:
                raise ExperimentError("cleanup could not confirm API absence and Firecracker exit")
            time.sleep(0.1)

    def run(self):
        started = time.monotonic()
        result = {
            "trial_id": self.id,
            "pair_index": int(self.id.split("-")[1]),
            "mode": self.mode,
            "success": False,
            "cycles": [],
            "validation_only": bool(getattr(self.args, "pause_before_cleanup", False)),
        }
        self.event("trial_started")
        try:
            self.context = self.node.prepare(self.id)
            result["clock_calibration"] = self.context.get("clock_calibration")
            self.event("sampler_started", node=self.context)
            self.measurement_started = time.monotonic()
            result["measurement_started_monotonic_seconds"] = self.measurement_started
            self.event("measurement_started")
            self.baseline_audit = self.node.audit()
            self.event("resource_audit_before", audit=self.baseline_audit)
            self.observe_node("before_create")
            self.lifecycle = "creating"
            create_started = time.monotonic()
            self.create_attempted = True
            _, created = self.api.request(
                "POST",
                "/sandboxes-cold",
                {
                    "image": self.args.guest_image,
                    "cpuCount": self.args.cpus,
                    "memoryMB": self.args.sandbox_memory_mib,
                    "timeout": self.ttl,
                    "secure": True,
                    "autoPause": False,
                    "autoResume": {"enabled": False},
                    "network": {"allowPublicTraffic": False},
                    "metadata": {"offload_run": self.id},
                },
                expected=(200, 201),
            )
            self.sid = created.get("sandboxID")
            if not self.sid:
                raise ExperimentError("create response lacks sandboxID")
            self.traffic_token = created.get("trafficAccessToken")
            if not self.traffic_token:
                raise ExperimentError("private application traffic token is missing")
            result["create_seconds"] = time.monotonic() - create_started
            result["sandbox_id"] = self.sid
            self.event("created", response=created)
            self.find_process()
            self.lifecycle = "running"
            result["ssh_aenv_bootstrap_seconds"] = self.node.launch_probe(self.sid)
            self.event("probe_bootstrap_returned", elapsed=result["ssh_aenv_bootstrap_seconds"])
            deadline = time.monotonic() + self.args.operation_timeout
            while True:
                try:
                    self.proxy("GET", "/health")
                    break
                except ExperimentError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            previous = self.proxy("GET", "/state")
            self.event("initial_oracle", state=previous)
            self.observe_node("ready")
            for cycle in range(self.args.cycles):
                self.refresh()
                idle_start = time.monotonic()
                self.event("idle_eligible", cycle=cycle)
                pause_seconds = self.pause() if self.mode == "offload" else 0.0
                # The simulated inference timer starts before pause; writeout
                # cost overlaps inference, rather than extending it artificially.
                # An explicit forecast may recover before the fixed tool
                # arrival. The request deadline never moves; early residency,
                # recovery and validation remain inside the measured task.
                requested = idle_start + self.args.wait_seconds
                lead = getattr(self.args, "resume_lead_seconds", 0.0) if self.mode == "offload" else 0.0
                self.event(
                    "tool_deadline_known",
                    cycle=cycle,
                    ready_monotonic_seconds=requested,
                    resume_lead_seconds=lead,
                    forecast_source="fixed synthetic inference wait",
                )
                time.sleep(max(0.0, requested - lead - time.monotonic()))
                recovery_started = time.monotonic()
                resume_seconds = self.resume() if self.mode == "offload" else 0.0
                restored = self.proxy("GET", "/state")
                assert_continuity(previous, restored)
                oracle_ready = time.monotonic()
                time.sleep(max(0.0, requested - time.monotonic()))
                self.event(
                    "tool_requested",
                    cycle=cycle,
                    ready_monotonic_seconds=requested,
                    observation_lag_seconds=max(0.0, time.monotonic() - requested),
                )
                idle_observation_seconds = 0.0
                oracle_seconds = max(0.0, oracle_ready - requested)
                state = self.tool(cycle)
                first_tool_seconds = time.monotonic() - requested
                if state["counter"] != previous["counter"] + 1:
                    raise OracleMismatch("tool counter did not advance exactly once")
                if state["fd_offset"] != min(previous["fd_offset"] + 17, previous["fd_size"]):
                    raise OracleMismatch("open file offset continuity failed")
                if state["nonce"] != previous["nonce"] or state["fd_sha256"] != previous["fd_sha256"]:
                    raise OracleMismatch("tool corrupted preserved runtime or file state")
                previous = state
                row = {
                    "cycle": cycle,
                    "pause_seconds": pause_seconds,
                    "resume_seconds": resume_seconds,
                    "restore_oracle_seconds": oracle_seconds,
                    "first_tool_seconds": first_tool_seconds,
                    "idle_actual_seconds": requested - idle_start,
                    "resume_lead_seconds": lead,
                    "recovery_started_monotonic_seconds": recovery_started,
                    "oracle_ready_monotonic_seconds": oracle_ready,
                    "recovery_to_oracle_seconds": oracle_ready - recovery_started,
                    "oracle_ready_before_tool_seconds": max(0.0, requested - oracle_ready),
                    "idle_end_ssh_observation_seconds": idle_observation_seconds,
                    **(self.last_resume_parts if self.mode == "offload" else {}),
                }
                result["cycles"].append(row)
                self.event("cycle_completed", **row)
            if getattr(self.args, "pause_before_cleanup", False):
                self.event("validation_pause_before_cleanup_requested")
                result["pause_before_cleanup_seconds"] = self.pause()
            result["success"] = True
            result["task_seconds"] = time.monotonic() - started
        except Exception as exc:
            result["error"] = clean(str(exc), self.api.secrets)
            result["error_type"] = type(exc).__name__
            result["correctness_failure"] = isinstance(exc, OracleMismatch)
            result["unknown_command"] = self.unknown_command
            result["task_seconds"] = time.monotonic() - started
            self.event("trial_failed", **result)
        finally:
            result["final_cleanup_state"] = self.lifecycle
            try:
                self.delete()
            except Exception as exc:
                self.cleanup = {"confirmed": False, "error": clean(str(exc), self.api.secrets)}
                self.event("cleanup_failed", **self.cleanup)
            self.cleanup["final_cleanup_state"] = result["final_cleanup_state"]
            if self.context is not None:
                try:
                    self.node.register(self.context, self.id, [])
                    time.sleep(
                        max(
                            self.args.io_tail_seconds,
                            (self.measurement_started or started) + self.args.observation_seconds - time.monotonic(),
                        )
                    )
                    result["measurement_finished_monotonic_seconds"] = time.monotonic()
                    self.event("measurement_finished")
                    time.sleep(2 * self.args.sample_interval)
                    self.node.finish(self.context, self.output)
                except Exception as exc:
                    result["metrics_error"] = clean(str(exc), self.api.secrets)
            result["cleanup"] = self.cleanup
            result["lifecycle_seconds"] = time.monotonic() - started
            self.event("trial_finished", result=result)
            (self.output / "result.json").write_text(
                json.dumps(clean(result, self.api.secrets), indent=2), encoding="utf-8"
            )
            (self.output / "cleanup.json").write_text(json.dumps(self.cleanup, indent=2), encoding="utf-8")
        return result


def summary(results: list[dict]) -> dict:
    output = {"started_trials": len(results), "by_mode": {}}
    for mode in ("resident", "offload"):
        group = [row for row in results if row["mode"] == mode]
        if group:
            successful = sum(row["success"] for row in group)
            cleaned = sum(bool(row["cleanup"].get("confirmed")) for row in group)
            output["by_mode"][mode] = {
                "started": len(group),
                "successful": successful,
                "success_rate": successful / len(group),
                "cleanup_confirmed": cleaned,
            }
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url")
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--node-ssh", required=True)
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--node-root", required=True)
    parser.add_argument("--aenv-bin", required=True)
    parser.add_argument("--serial-dir", required=True)
    parser.add_argument("--state-dir", default="/opt/agentenv-offload/state")
    parser.add_argument("--runtime-dir", default="/run/aenv")
    parser.add_argument("--provider-pid", type=int, action="append", default=[])
    parser.add_argument("--provider-pid-file")
    parser.add_argument("--guest-image", required=True, help="Pinned image@sha256:... or AgentEnv sha256:... image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("resident", "offload", "both"), default="both")
    parser.add_argument(
        "--pause-before-cleanup",
        action="store_true",
        help="Validation-only: pause after completed cycles, then delete the paused sandbox",
    )
    parser.add_argument("--wait-seconds", type=float, default=30)
    parser.add_argument(
        "--resume-lead-seconds",
        type=float,
        default=0.0,
        help="Explicit fixed-wait forecast: begin recovery this long before tool arrival",
    )
    parser.add_argument("--memory-mib", type=int, default=64)
    parser.add_argument("--sandbox-memory-mib", type=int, default=1024)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--entropy", choices=("low", "high"), default="low")
    parser.add_argument("--dirty-ratio", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--operation-timeout", type=float, default=180)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--io-tail-seconds", type=float, default=2)
    parser.add_argument(
        "--observation-seconds",
        type=float,
        default=0,
        help="Minimum elapsed trial window to sample, including post-cleanup idle",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.resume_lead_seconds) or not 0 <= args.resume_lead_seconds <= args.wait_seconds:
        parser.error("resume-lead-seconds must be finite and within 0..wait-seconds")
    if args.pause_before_cleanup and args.mode != "offload":
        parser.error("--pause-before-cleanup is validation-only and requires --mode offload")
    if not re.search(r"(?:^|@)sha256:[0-9a-f]{64}$", args.guest_image):
        parser.error("--guest-image must use a fixed digest")
    if min(args.memory_mib, args.sandbox_memory_mib, args.cycles, args.pairs, args.cpus) <= 0:
        parser.error("memory, cycles, pairs and cpus must be positive")
    if args.memory_mib + 128 >= args.sandbox_memory_mib:
        parser.error("leave more than 128 MiB beyond probe memory for the guest runtime")
    if (
        not all(
            math.isfinite(value) and value >= 0
            for value in (args.wait_seconds, args.dirty_ratio, args.io_tail_seconds, args.observation_seconds)
        )
        or args.dirty_ratio > 1
    ):
        parser.error("wait/tail must be finite and nonnegative; dirty-ratio must be in 0..1")
    if not all(math.isfinite(value) and value > 0 for value in (args.operation_timeout, args.sample_interval)):
        parser.error("timeouts and sample interval must be positive and finite")
    for path in (args.node_root, args.aenv_bin, args.serial_dir, args.state_dir, args.runtime_dir):
        if not path.startswith("/") or ".." in Path(path).parts:
            parser.error("node paths must be absolute without parent traversal")
    if not args.provider_pid and not args.provider_pid_file:
        parser.error("provide the node's provider/ublk process registry")
    return args


def main(argv=None):
    args = parse_args(argv)
    credentials = read_credentials(args.credentials_file)
    api = API(args.api_url or credentials.get("url", ""), credentials["api_key"], args.operation_timeout)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    source_hashes = archive_sources(args.output)
    manifest = {
        "schema_version": 1,
        "required_agentenv_sha": AGENTENV_SHA,
        "created_ns": time.time_ns(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "credentials_file"
        },
        "source_sha256": source_hashes,
        "source_archive": "source",
        "validation_only": bool(args.pause_before_cleanup),
        "control_path": {
            "version": 3,
            "fixed_wait_forecast": True,
            "resume_lead_seconds": getattr(args, "resume_lead_seconds", 0.0),
            "tool_deadline_unchanged": True,
            "resume_requires_http_status": 201,
            "resume_discovery_registration_ssh_calls": 1,
            "idle_end_synchronous_ssh": False,
            "full_memory_oracle": True,
        },
        "credential_source": "private TOML file",
        "workload": "external simulated inference wait",
        "tool_workload": "HTTP probe file/counter mutation; aenv CLI used for guest bootstrap",
        "deployment_revision_validation": "requires deployment manifest; not inferred from API health",
        "limitations": ["single node", "no cross-node recovery", "no workload success-rate SLA claim"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    node = Node(args)
    results = []
    generator = random.Random(args.seed)
    for pair in range(args.pairs):
        modes = ["resident", "offload"] if args.mode == "both" else [args.mode]
        generator.shuffle(modes)
        for mode in modes:
            trial_id = f"pair-{pair:03d}-{mode}-{uuid.uuid4().hex[:8]}"
            result = Trial(args, api, node, mode, trial_id).run()
            results.append(result)
            with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(clean(result, api.secrets)) + "\n")
            report = summary(results)
            (args.output / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "success": result["success"],
                        "cleanup_confirmed": result["cleanup"].get("confirmed", False),
                    }
                ),
                flush=True,
            )
            if (
                not result["success"]
                or not result["cleanup"].get("confirmed")
                or result.get("unknown_command")
                or result.get("correctness_failure")
            ):
                return 1
    return 0 if all(row["success"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
