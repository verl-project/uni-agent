import ctypes
import os
import signal
import subprocess
import sys
import time

import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_encoder import SharedEncoderActor, ensure_ray_initialized

NUM_RESERVED_CPUS = int(os.getenv("NUM_RESERVED_CPUS", "16"))
NUM_ENCODER_GPUS = int(os.getenv("NUM_ENCODER_GPUS", "1"))
data_root = os.getenv("DATA_ROOT", os.path.expanduser("~/uni_agent_data"))


def _preexec_die_with_parent():
    """Run in the uvicorn subprocess: detach into its own process group so we
    can kill the whole tree, and ask the kernel to SIGTERM us if the actor
    process (our parent) ever dies (Linux-only, via prctl PR_SET_PDEATHSIG).
    """
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        pass


@ray.remote(num_cpus=NUM_RESERVED_CPUS)
class WikiRetrievalManager:
    def __init__(self, config):
        self.config = config
        self.process = None

    def start_service(self):
        env = os.environ.copy()

        cmd = [
            "python",
            "-m",
            "uvicorn",
            "retrieval_server:app",
            "--host",
            "::",
            "--port",
            str(self.config["port"]),
            "--timeout-keep-alive",
            "600",
        ]

        env.update(
            {
                "INDEX_PATH": self.config["index_file"],
                "CORPUS_PATH": self.config["corpus_file"],
                "RETRIEVER_NAME": self.config["retriever_name"],
                "RETRIEVER_MODEL": self.config["retriever_path"],
                "TOPK": str(self.config["topk"]),
                "BATCH_SIZE": str(self.config["batch_size"]),
                "HTTP_MAX_REQUEST_BATCH_SIZE": str(self.config["http_max_request_batch_size"]),
                "HTTP_BATCH_TIMEOUT": str(self.config["http_batch_timeout_s"]),
                "RAY_ADDRESS": self.config["ray_address"],
                "RAY_NAMESPACE": self.config["ray_namespace"],
                "SHARED_ENCODER_ACTOR_NAME": self.config["shared_encoder_actor_name"],
                "NUM_ENCODER_GPUS": str(self.config["num_encoder_gpus"]),
            }
        )

        print(f"Starting Wiki Retrieval HTTP service on port {self.config['port']}")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            preexec_fn=_preexec_die_with_parent,
        )

    def stop_service(self):
        """Best-effort kill of the uvicorn process group."""
        if self.process is None:
            return
        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.process = None

    def __del__(self):
        try:
            self.stop_service()
        except Exception:
            pass

    def wait_forever(self):
        while True:
            retcode = self.process.wait()
            print(f"Service exited with code {retcode}, restarting...")
            time.sleep(2)
            self.start_service()


def _encoder_actor_names(config):
    base_name = config["shared_encoder_actor_name"]
    num_gpus = config["num_encoder_gpus"]
    if num_gpus == 1:
        return [base_name]
    return [f"{base_name}-{i}" for i in range(num_gpus)]


def get_or_create_shared_encoders(config):
    """Create or reuse shared encoder actor(s). Returns a list of actor handles."""
    namespace = config["ray_namespace"]
    recreate = config["recreate_shared_encoder"]
    actor_names = _encoder_actor_names(config)

    actors = []
    for name in actor_names:
        try:
            actor = ray.get_actor(name, namespace=namespace)
            if recreate:
                print(f"Replacing existing shared encoder actor: {name}")
                ray.kill(actor, no_restart=True)
            else:
                print(f"Reusing existing shared encoder actor: {name}")
                actors.append(actor)
                continue
        except ValueError:
            pass

        print(f"Creating shared encoder actor: {name}")
        actor = SharedEncoderActor.options(
            name=name,
            lifetime="detached",
        ).remote(
            model_name=config["retriever_name"],
            model_path=config["retriever_path"],
            pooling_method=config["retrieval_pooling_method"],
            max_length=config["retrieval_query_max_length"],
            use_fp16=config["retrieval_use_fp16"],
            max_batch_size=config["encoder_max_batch_size"],
            batch_timeout_s=config["encoder_batch_timeout_s"],
        )
        actors.append(actor)

    return actors


ensure_ray_initialized()


wiki_config = {
    "index_file": f"{data_root}/wiki24/wiki24_faiss.index",
    "corpus_file": f"{data_root}/wiki24/wiki24_preprocessed/",
    "retriever_name": "bge-m3",
    "retriever_path": f"{data_root}/model/bge-m3",
    "port": 8001,
    "topk": 3,
    "batch_size": int(os.getenv("RETRIEVER_BATCH_SIZE", "512")),
    "http_max_request_batch_size": int(os.getenv("HTTP_MAX_REQUEST_BATCH_SIZE", "64")),
    "http_batch_timeout_s": float(os.getenv("HTTP_BATCH_TIMEOUT", "0.002")),
    "ray_address": os.getenv("RAY_ADDRESS", "auto"),
    "ray_namespace": os.getenv("RAY_NAMESPACE", "localwiki"),
    "num_encoder_gpus": NUM_ENCODER_GPUS,
    "shared_encoder_actor_name": os.getenv("SHARED_ENCODER_ACTOR_NAME", "localwiki-shared-encoder"),
    "retrieval_pooling_method": "mean",
    "retrieval_query_max_length": 512,
    "retrieval_use_fp16": True,
    "encoder_max_batch_size": int(os.getenv("SHARED_ENCODER_MAX_BATCH_SIZE", "2048")),
    "encoder_batch_timeout_s": float(os.getenv("SHARED_ENCODER_BATCH_TIMEOUT", "0.008")),
    "recreate_shared_encoder": os.getenv("RECREATE_SHARED_ENCODER", "true").lower() in ("true", "1", "t"),
}

encoder_actors = get_or_create_shared_encoders(wiki_config)
ray.get([actor.encode.remote(["localwiki warmup"]) for actor in encoder_actors])
print(f"All {len(encoder_actors)} encoder actor(s) warmed up.")

wiki_actor = WikiRetrievalManager.remote(wiki_config)
wiki_actor.start_service.remote()

try:
    ray.get(wiki_actor.wait_forever.remote())
except KeyboardInterrupt:
    print("Driver received Ctrl+C, stopping wiki service...")
    try:
        ray.get(wiki_actor.stop_service.remote(), timeout=15)
    except Exception as e:
        print(f"stop_service failed: {e}")
    ray.kill(wiki_actor, no_restart=True)
