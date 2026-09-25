from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from uni_agent.sandbox import ImageMount, SandboxConfig, build_sandbox


@pytest.mark.cpu
@pytest.mark.level0
def test_image_mount_config_is_provider_independent():
    config = SandboxConfig(
        provider="modal",
        image="example/task:latest",
        image_mounts=[
            {
                "image": "example/claude-code:latest",
                "mount_path": "/opt/claude-code/",
            }
        ],
    )

    assert config.image_mounts == [
        ImageMount(
            image="example/claude-code:latest",
            mount_path="/opt/claude-code",
        )
    ]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("mount_path", ["relative/path", "/", "/opt/../agent"])
def test_image_mount_path_must_be_absolute_non_root_and_safe(mount_path):
    with pytest.raises(ValueError, match="mount_path"):
        ImageMount(image="example/agent:latest", mount_path=mount_path)


@pytest.mark.cpu
@pytest.mark.level0
def test_image_mount_paths_must_be_unique():
    with pytest.raises(ValueError, match="unique mount_path"):
        SandboxConfig(
            provider="modal",
            image="example/task:latest",
            image_mounts=[
                {"image": "example/agent-a:latest", "mount_path": "/opt/agent"},
                {"image": "example/agent-b:latest", "mount_path": "/opt/agent/"},
            ],
        )


@pytest.mark.cpu
@pytest.mark.level0
def test_local_provider_rejects_container_image():
    config = SandboxConfig(provider="local", image="example/task:latest")
    with pytest.raises(ValueError, match="LocalSandbox does not accept an image"):
        build_sandbox(config)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("provider", "image"),
    [
        ("local", None),
        ("vefaas", "python:3.12"),
    ],
)
def test_registry_rejects_image_mounts_for_unsupported_provider(provider, image):
    config = SandboxConfig(
        provider=provider,
        image=image,
        image_mounts=[{"image": "example/agent:latest", "mount_path": "/opt/agent"}],
    )

    with pytest.raises(NotImplementedError, match="does not support image mounts"):
        build_sandbox(config)


@pytest.mark.cpu
@pytest.mark.level0
def test_unsupported_provider_constructor_rejects_image_mounts():
    from uni_agent.sandbox.vefaas import VefaasSandbox

    mount = ImageMount(image="example/agent:latest", mount_path="/opt/agent")
    with pytest.raises(NotImplementedError, match="does not support image mounts"):
        VefaasSandbox(image_mounts=[mount])


@pytest.mark.cpu
@pytest.mark.level0
def test_supported_provider_constructor_accepts_image_mounts():
    from uni_agent.sandbox.docker import DockerSandbox
    from uni_agent.sandbox.modal import ModalSandbox
    from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox

    mount = ImageMount(image="example/agent:latest", mount_path="/opt/agent")
    for sandbox in [
        DockerSandbox(image="example/task:latest", image_mounts=[mount]),
        ModalSandbox(image_mounts=[mount]),
        OpenyuanrongSandbox(image="example/task:latest", image_mounts=[mount]),
    ]:
        assert sandbox.image_mounts == [mount]


@pytest.mark.cpu
@pytest.mark.level0
def test_openyuanrong_maps_image_mounts_to_sdk_mounts(monkeypatch):
    from uni_agent.sandbox import openyuanrong

    created: dict = {}

    class _Mount:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Sandbox:
        def __init__(self, **kwargs):
            created.update(kwargs)

    sdk = SimpleNamespace(Mount=_Mount, Sandbox=_Sandbox)
    monkeypatch.setattr(openyuanrong, "_load_sdk", lambda: sdk)

    config = SandboxConfig(
        provider="openyuanrong",
        image="example/task:latest",
        image_mounts=[
            {"image": "example/claude-code:latest", "mount_path": "/opt/claude-code"},
            {"image": "example/verifier:latest", "mount_path": "/opt/verifier"},
        ],
        sandbox_kwargs={
            "connection": object(),
            "mounts": [{"image_url": "example/data:latest", "target": "/data"}],
        },
    )
    sandbox = build_sandbox(config)

    asyncio.run(sandbox.start())

    assert sandbox.image_mounts == config.image_mounts
    assert [mount.kwargs for mount in created["mounts"]] == [
        {"target": "/data", "image_url": "example/data:latest"},
        {"target": "/opt/claude-code", "image_url": "example/claude-code:latest"},
        {"target": "/opt/verifier", "image_url": "example/verifier:latest"},
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_modal_builds_and_mounts_each_image(monkeypatch):
    calls: list[tuple] = []
    app = SimpleNamespace(app_id="ap-1")

    class _AioMethod:
        def __init__(self, fn):
            self.aio = fn

    class _Image:
        def __init__(self, ref: str):
            self.ref = ref
            self.build = _AioMethod(self._build)

        async def _build(self, actual_app):
            calls.append(("build", self.ref, actual_app))
            return self

    class _ImageFactory:
        @staticmethod
        def from_registry(ref: str):
            calls.append(("from_registry", ref))
            return _Image(ref)

    class _RunningSandbox:
        def __init__(self):
            self.mount_image = _AioMethod(self._mount_image)

        async def _mount_image(self, path, image):
            calls.append(("mount_image", path, image.ref))

    running = _RunningSandbox()

    async def _lookup(name, *, create_if_missing):
        calls.append(("lookup", name, create_if_missing))
        return app

    async def _create(*args, **kwargs):
        calls.append(("create", args, kwargs))
        return running

    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=_AioMethod(_lookup)),
        Image=_ImageFactory,
        Sandbox=SimpleNamespace(create=_AioMethod(_create)),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)

    config = SandboxConfig(
        provider="modal",
        image="example/task:latest",
        image_mounts=[
            {"image": "example/claude-code:latest", "mount_path": "/opt/claude-code"},
            {"image": "example/verifier:latest", "mount_path": "/opt/verifier"},
        ],
    )
    sandbox = build_sandbox(config)

    asyncio.run(sandbox.start())

    assert sandbox.image_mounts == config.image_mounts
    assert [call for call in calls if call[0] == "build"] == [
        ("build", "example/claude-code:latest", app),
        ("build", "example/verifier:latest", app),
    ]
    assert [call for call in calls if call[0] == "mount_image"] == [
        ("mount_image", "/opt/claude-code", "example/claude-code:latest"),
        ("mount_image", "/opt/verifier", "example/verifier:latest"),
    ]
