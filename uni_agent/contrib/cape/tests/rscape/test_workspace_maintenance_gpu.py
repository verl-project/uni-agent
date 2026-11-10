# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from pycape.workspace_fs import mount_entries
from tests.rscape import test_functional_parity as parity
from tests.workspace_fixtures import copy_workspace_evidence


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_MAINTENANCE_GPU_UUID"), "requires an explicitly authorized GPU UUID")
class WorkspaceMaintenanceGpuTests(unittest.TestCase):
    def test_maintenance_releases_gpu_for_unrelated_command(self):
        gpu=os.environ["CAPE_WORKSPACE_MAINTENANCE_GPU_UUID"]
        helper=parity.FunctionalParityTests()
        for binary in (parity.PYCAPE_BIN,parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                root=Path(tempfile.mkdtemp(prefix="cape-maintenance-gpu-"))
                controller=node=None;passed=False
                flag,entered,unblock,marker=(root/name for name in ("gate","entered","unblock","gpu-started"))
                try:
                    controller,url,_=helper.start_controller(binary,str(root),"--idle-session-timeout-seconds","180")
                    helper.request("POST",url,"/v1/admin/workspace-config",{"pool":"maint","changes":{
                        "image_root":str(root/"images"),"image_size_bytes":64*1024**2,
                        "maintenance":{"auto_grow":True,"grow_threshold_percent":60,"grow_target_percent":40,
                                       "grow_min_bytes":64*1024**2,"backing_reserve_bytes":0,"timeout_seconds":60}}})
                    wrapper=root/"copy.sh"
                    wrapper.write_text('#!/bin/sh\nif [ -e "'+str(flag)+'" ]; then\n'
                                       'printf entered > "'+str(entered)+'"\n'
                                       'while [ ! -e "'+str(unblock)+'" ]; do sleep 0.05; done\nfi\nexec /usr/bin/cp "$@"\n')
                    wrapper.chmod(0o700)
                    with (root/"node.log").open("w") as output:
                        node=subprocess.Popen([binary,"node", "enroll", *helper.node_credentials(url, "node", "maint"),"--controller-url",url,"--pool","maint","--node-id", "node",
                            "--runtime-adapter","local-process","--gpu-uuid",gpu,"--cpu-cores","4","--memory-gb","8",
                            "--heartbeat-interval","0.05","--state-path",str(root/"state.json"),"--log-dir",str(root/"logs")],
                            env=dict(os.environ,CAPE_WORKSPACE_CP_BIN=str(wrapper)),stdout=output,stderr=output)
                    deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        self.assertIsNone(node.poll(),(root/"node.log").read_text()[-4000:])
                        rows=helper.request("GET",url,"/v1/admin/workspace-config?pool=maint")["nodes"]
                        if rows and rows[0]["state"]=="applied":break
                        time.sleep(.05)
                    else:self.fail("node readiness timeout")
                    flag.touch()
                    def submit(user,task,code,identity=None):
                        body={"pool":"maint","user_id":user,"session_id":task,"reuse":True,
                              "command":["python3","-c",code],"resources":{"gpus":1,"runtime_adapter":"local-process","isolation_policy":"none"}}
                        if identity:body["workspace"]={"id":identity}
                        return helper.request("POST",url,"/v1/submit",body)
                    cuda="import torch; assert torch.cuda.device_count()==1; x=torch.ones(128,device='cuda'); assert x.sum().item()==128; torch.cuda.synchronize(); "
                    first=submit("alice","a","from pathlib import Path; import time; Path('payload').write_bytes(b'x'*(40*1024**2)); "+cuda+"time.sleep(2)")
                    active=helper.wait_state(url,first["request_id"],"RUNNING",timeout=90)
                    identity=active["workspace"]["id"]
                    second=submit("alice","a","from pathlib import Path; assert Path('payload').stat().st_size==40*1024**2; "+cuda,identity)
                    other=submit("bob","b","from pathlib import Path; import time; "+cuda+"Path("+repr(str(marker))+").write_text('cuda-passed'); time.sleep(2)")
                    deadline=time.monotonic()+60
                    while time.monotonic()<deadline and not (entered.exists() and marker.exists()):time.sleep(.02)
                    self.assertTrue(entered.exists() and marker.exists(),"unrelated CUDA work did not run during held maintenance")
                    self.assertEqual(helper.request("GET",url,"/v1/workspaces/"+identity)["operation"]["state"],"pending")
                    self.assertEqual(helper.request("GET",url,"/v1/requests/"+first["request_id"])["state"],"COMPLETED")
                    self.assertEqual(helper.request("GET",url,"/v1/requests/"+second["request_id"])["state"],"QUEUED")
                    self.assertEqual(helper.request("GET",url,"/v1/requests/"+other["request_id"])["state"],"RUNNING")
                    self.assertEqual(helper.request("GET",url,"/v1/pools/maint/utilization")["physical"]["exclusive_leases"],1)
                    unblock.touch()
                    helper.wait_state(url,other["request_id"],"COMPLETED",timeout=90)
                    done=helper.wait_state(url,second["request_id"],"COMPLETED",timeout=90)
                    self.assertEqual(done["workspace"]["id"],identity)
                    passed=True
                    print(json.dumps({"binary":binary,"gpu":gpu,"unrelated_cuda_during_maintenance":"passed",
                                      "same_workspace_gpu_reacquisition":"passed"}),flush=True)
                finally:
                    unblock.touch()
                    for process in (node,controller):
                        if process is not None and process.poll() is None:
                            process.terminate()
                            try:process.wait(timeout=30)
                            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                        if process is not None:
                            for stream in (process.stdin,process.stdout,process.stderr):
                                if stream is not None:stream.close()
                    output=os.environ.get("CAPE_WORKSPACE_MAINTENANCE_OUTPUT")
                    if output:
                        dest=Path(output)/("gpu-"+Path(binary).name)
                        copy_workspace_evidence(root,dest)
                    if any(row["target"].startswith(str(root)+"/") for row in mount_entries()):raise AssertionError("GPU maintenance mounts remain")
                    if passed:shutil.rmtree(root)
