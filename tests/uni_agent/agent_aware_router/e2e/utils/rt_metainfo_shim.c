/* Load shim for vllm_ascend's libvllm_ascend_kernels.so on the gate's CANN 9.2
 * image. The kernel wheel imports two symbols that image does not export:
 * rtFunctionGetMetaInfoSize and MsprofUnRegisterCallback. dlopen fails on the
 * first missing symbol, which blocks custom-op registration and the NPU worker.
 *
 * rtFunctionGetMetaInfoSize's only caller is the DeepSeek-style MLA preprocess
 * path, unused by the Qwen3 models this e2e runs; it aborts rather than
 * fabricating a size, so a model that reaches it fails loudly instead of
 * silently corrupting MLA state. MsprofUnRegisterCallback is a profiling hook;
 * a no-op stub just leaves the profiler unregistered.
 *
 *   gcc -shared -fPIC -O2 -o librt_metainfo_shim.so rt_metainfo_shim.c
 *   LD_PRELOAD=$PWD/librt_metainfo_shim.so python ...
 */
#include <stdio.h>
#include <stdlib.h>

int rtFunctionGetMetaInfoSize(void *a, void *b, void *c, void *d)
{
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    fprintf(stderr, "[rt_metainfo_shim] rtFunctionGetMetaInfoSize was called: MLA path reached, real CANN required.\n");
    abort();
}

int MsprofUnRegisterCallback(int callbackId)
{
    (void)callbackId;
    return 0;
}
