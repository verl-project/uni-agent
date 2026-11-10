# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from .models import MountSpec, ResourceRequest, WorkloadSpec


class EnvironmentProfile(object):
    def __init__(self, name, image=None, default_mounts=(), runtime_preference=None, gpu_required=False, env=None):
        self.name = name
        self.image = image
        self.default_mounts = tuple(default_mounts or ())
        self.runtime_preference = runtime_preference
        self.gpu_required = gpu_required
        self.env = dict(env or {})

    @classmethod
    def from_dict(cls, name, data):
        mounts = []
        for item in data.get("default_mounts", ()):
            if isinstance(item, MountSpec):
                mounts.append(item)
            elif isinstance(item, str):
                parts = item.split(":")
                if len(parts) < 2:
                    raise ValueError("mount string must be source:target[:ro|rw]")
                mounts.append(MountSpec(parts[0], parts[1], len(parts) > 2 and parts[2] == "ro"))
            else:
                mounts.append(MountSpec(item["source"], item["target"], item.get("read_only", False)))
        return cls(
            name=name,
            image=data.get("image"),
            default_mounts=tuple(mounts),
            runtime_preference=data.get("runtime_preference"),
            gpu_required=data.get("gpu_required", False),
            env=data.get("env", {}),
        )


class ProfileRegistry(object):
    def __init__(self, profiles=None):
        self.profiles = {}
        for name, profile in dict(profiles or {}).items():
            if isinstance(profile, EnvironmentProfile):
                self.profiles[name] = profile
            else:
                self.profiles[name] = EnvironmentProfile.from_dict(name, profile)

    @classmethod
    def from_json_file(cls, path):
        with open(path) as fh:
            data = json.load(fh)
        return cls(data.get("profiles", data))

    def get(self, name):
        if name not in self.profiles:
            raise KeyError("unknown environment profile: " + name)
        return self.profiles[name]

    def resolve(self, spec):
        if not spec.profile:
            return spec
        profile = self.get(spec.profile)
        resources = spec.resources
        gpus = resources.gpus
        if profile.gpu_required and gpus == 0:
            gpus = 1
        runtime_adapter = resources.runtime_adapter or profile.runtime_preference
        resolved_resources = ResourceRequest(
            gpus=gpus,
            cpu_cores=resources.cpu_cores,
            memory_gb=resources.memory_gb,
            max_duration_seconds=resources.max_duration_seconds,
            command_timeout_seconds=resources.command_timeout_seconds,
            hard_gpu_enforcement=resources.hard_gpu_enforcement,
            runtime_adapter=runtime_adapter,
            ports=resources.ports,
            locality=resources.locality,
            isolation_policy=resources.isolation_policy,
        )
        env = dict(profile.env)
        env.update(spec.env)
        resolved = WorkloadSpec(
            pool=spec.pool,
            user_id=spec.user_id,
            session_id=spec.session_id,
            command=spec.command,
            resources=resolved_resources,
            cwd=spec.cwd,
            env=env,
            image=spec.image or profile.image,
            profile=None,
            mounts=tuple(profile.default_mounts) + tuple(spec.mounts),
            reuse=spec.reuse,
            keep_session_on_failure=spec.keep_session_on_failure,
            priority=spec.priority,
            privileged=spec.privileged,
            log_path=spec.log_path,
            artifact_paths=spec.artifact_paths,
            workspace=spec.workspace,
        )
        resolved.resolved_profile = spec.profile
        return resolved
