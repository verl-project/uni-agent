# ARM64 Compatibility Notes

Uni-Agent uses the same Python installation and sandbox workflows on x86_64 and
ARM64. Follow [Installation](installation.md) and the [uv environment](uv.md)
guides. For CPU-only development, select the `cpu` extra; agents can use an
external OpenAI-compatible endpoint, so local model inference is not required.

This page covers architecture-specific Docker task and tool image issues.
Native ARM64 builds are recommended when available; amd64 emulation is a
compatibility fallback. Native ARM64 CPU workflows were verified on openEuler
24.03 with Python 3.11 and Docker 25.0.3.

## Check Image and Binary Architectures

On an ARM64 host, `uname -m` returns `aarch64`. Check every task and sidecar
image before starting a sandbox:

```bash
docker image inspect <image>:<tag> --format '{{.Os}}/{{.Architecture}}'
```

Use `linux/arm64` for native execution. A `linux/amd64` image requires emulation
on an ARM64 host.

Image metadata does not guarantee the architecture of binaries copied into the
image. For example, a multi-stage sidecar can contain an x86_64 Python runtime
even when the final image is labeled `linux/arm64`. Test the actual tool
entrypoint, replacing the path and version flag with those for your tool:

```bash
docker run --rm --platform linux/arm64 \
  --entrypoint /opt/my-tool/bin/my-tool \
  my-tool:arm64 --version
```

An `exec format error` usually means that the executable does not match the
selected platform or that the required emulator is unavailable.

## Build a Native ARM64 Sidecar

Use ARM64 base images, runtimes, and tool artifacts throughout the build:

```bash
docker build --platform linux/arm64 \
  -f Dockerfile.my-tool -t my-tool:arm64 .
docker image inspect my-tool:arm64 --format '{{.Os}}/{{.Architecture}}'
```

If the recipe's build script hard-codes `--platform linux/amd64`, override or
change that setting. Also check every downloaded archive and prebuilt runtime;
changing the Docker platform alone does not make an x86_64 binary run natively.

BuildKit provides `TARGETARCH` for selecting the correct artifact. Declare it in
the build stage that downloads or installs the tool:

```dockerfile
ARG TARGETARCH
RUN case "$TARGETARCH" in \
      amd64) TOOL_ARCH=x86_64 ;; \
      arm64) TOOL_ARCH=aarch64 ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac && \
    echo "Download and install the ${TOOL_ARCH} artifact here"
```

Replace the final `echo` with the tool's download and installation commands.
If the tool publishes only amd64 binaries, use emulation or choose a version
with ARM64 support.

Keep the sidecar's root layout and configure `image_mounts` and
`executable_paths` as described in
[Build a Tool Image](launch-sandbox.md#build-a-tool-image-sidecar).

## Use amd64 Emulation as a Fallback

When a native ARM64 build is unavailable, register amd64 emulation through
QEMU and binfmt:

```bash
docker run --privileged --rm tonistiigi/binfmt --install amd64
docker run --rm --platform linux/amd64 alpine:3.22 uname -m
```

The second command should print `x86_64`. The installer modifies the host's
binfmt handlers and requires privileged access.

Build or pull the amd64 sidecar and confirm its platform:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.my-tool -t my-tool:amd64 .
docker image inspect my-tool:amd64 --format '{{.Os}}/{{.Architecture}}'
```

Use that image in `image_mounts`. An ARM64 task container mounting an amd64
sidecar still needs working amd64 emulation to execute the mounted binaries.
Keep the task image and sidecar on the same architecture when possible.

Emulation is slower than native execution, and tools that depend on
architecture-specific instructions or unsupported system calls may still fail.
Prefer native ARM64 images for repeated evaluations and production workflows.

!!! note "Local model inference"
    Local inference is not required for CPU-only development. vLLM can be built
    from source on ARM64 hosts with supported NVIDIA GPUs; that setup depends
    on the GPU and CUDA stack and is outside the verified environment above.
