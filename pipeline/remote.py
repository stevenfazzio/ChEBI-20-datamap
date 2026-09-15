"""Run stage 02 (and 04) on a Runpod GPU pod: provision, sync code + inputs, run, pull results, tear down.

Lane: runpodctl (non-interactive; it reads RUNPOD_API_KEY or ~/.runpod/config.toml) plus plain ssh/scp
with the account's registered key. The pod is a throwaway: it is deleted in a `finally` unless
`keep_pod` is set, because this runpodctl build has no --terminate-after guard.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import config

SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ServerAliveInterval=30",
]


class RunpodError(RuntimeError):
    pass


def runpodctl(*args: str, timeout: float = 900):
    """Call runpodctl and parse its JSON. Failures are a JSON object on stderr; surface code + message."""
    proc = subprocess.run(["runpodctl", *args, "-o", "json"], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        try:
            err = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            err = {"error": proc.stderr.strip() or proc.stdout.strip(), "code": "unknown"}
        raise RunpodError(f"runpodctl {' '.join(args)} -> [{err.get('code')}] {err.get('error')}")
    out = proc.stdout.strip()
    return json.loads(out) if out else None


def _pods(listing) -> list[dict]:
    if isinstance(listing, list):
        return listing
    if isinstance(listing, dict):
        return next((v for v in listing.values() if isinstance(v, list)), [])
    return []


def find_pod() -> dict | None:
    for pod in _pods(runpodctl("pod", "list")):
        status = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
        if pod.get("name") == config.RUNPOD_POD_NAME and status == "RUNNING":
            return pod
    return None


_STOCK_RANK = {"High": 3, "Medium": 2, "Low": 1}


def placements(max_per_gpu: int = 2) -> list[tuple[str, str, str]]:
    """(gpu_id, data_center_id, stock) for each candidate GPU, best-stocked data centers first."""
    listing = runpodctl("gpu", "list", "--include-unavailable")
    items = listing if isinstance(listing, list) else next(v for v in listing.values() if isinstance(v, list))
    by_id = {g.get("gpuId"): g for g in items}
    out = []
    for gpu_id in config.RUNPOD_GPU_CANDIDATES:
        g = by_id.get(gpu_id)
        if not g:
            continue
        dcs = [
            (d.get("dataCenterId"), d.get("stockStatus"))
            for d in g.get("dataCenterAvailability") or []
            if d.get("stockStatus") in _STOCK_RANK
        ]
        dcs.sort(key=lambda x: -_STOCK_RANK[x[1]])
        out += [(gpu_id, dc, stock) for dc, stock in dcs[:max_per_gpu]]
    return out


def _create_pod_once(gpu_id: str, data_center: str, wait_timeout: str) -> dict:
    return runpodctl(
        "pod",
        "create",
        "--name",
        config.RUNPOD_POD_NAME,
        "--template-id",
        config.RUNPOD_TEMPLATE_ID,
        "--gpu-id",
        gpu_id,
        "--data-center-ids",
        data_center,
        "--cloud-type",
        config.RUNPOD_CLOUD_TYPE,
        "--ports",
        "22/tcp",
        "--container-disk-in-gb",
        str(config.RUNPOD_CONTAINER_DISK_GB),
        "--wait",
        "--wait-timeout",
        wait_timeout,
        timeout=1200,
    )


def create_pod(wait_timeout: str = "4m") -> dict:
    """Create the pod in a data center that reports stock, walking the candidate GPUs in order.

    A healthy pod answers ssh in ~20 s; one without an ssh port after a few minutes never will, so on
    `wait_timeout` the orphan is deleted and the next placement is tried. Pods bill by the second."""
    options = placements()
    if not options:
        raise RunpodError("no candidate GPU reports stock in any data center right now")
    print(f"placements with stock: {[(g.split()[-1], dc, s) for g, dc, s in options]}", flush=True)
    last = None
    for gpu_id, dc, stock in options:
        print(
            f"creating pod {config.RUNPOD_POD_NAME!r}: {gpu_id} in {dc} (stock {stock}), waiting for ssh...", flush=True
        )
        try:
            pod = _create_pod_once(gpu_id, dc, wait_timeout)
            pod["_gpu_id"], pod["_data_center"] = gpu_id, dc
            return pod
        except RunpodError as e:
            last = e
            if "[wait_timeout]" in str(e):
                m = re.search(r"pod (\w+) was created", str(e))
                if m:
                    runpodctl("pod", "delete", m.group(1))
                    print(f"  pod {m.group(1)} never got an ssh port in {wait_timeout}; deleted it", flush=True)
            elif (
                "[bad_request]" in str(e)
                or "[conflict]" in str(e)
                or "not available" in str(e).lower()
                or "instances available" in str(e).lower()  # graphql_error: stock vanished between list and create
            ):
                print(f"  {gpu_id} in {dc}: {str(e).split('->')[-1].strip()[:120]}", flush=True)
            else:
                raise
    raise RunpodError(f"no placement came up: {last}")


DEFAULT_SSH_KEY = Path.home() / ".ssh" / "id_ed25519"  # the key registered with the account


def ssh_target(pod_id: str) -> dict:
    """runpodctl 2.12 `ssh info` shape: {ip, port, ssh_command: "ssh root@IP -p PORT", ssh_key: {path, exists}}."""
    info = runpodctl("ssh", "info", pod_id)
    cmd = str(info.get("ssh_command") or info.get("command") or "")
    m_user_host = re.search(r"([\w.-]+)@([\w.-]+)", cmd)
    user = m_user_host.group(1) if m_user_host else "root"
    host = (m_user_host.group(2) if m_user_host else None) or info.get("ip")
    m_port = re.search(r"-p\s+(\d+)", cmd)
    port = m_port.group(1) if m_port else info.get("port")
    if not (host and port):
        raise RunpodError(f"could not parse ssh info for {pod_id}: {info}")
    key_info = info.get("ssh_key") or {}
    key = key_info.get("path") if key_info.get("exists") else None
    if not key:
        if not DEFAULT_SSH_KEY.exists():
            raise RunpodError(f"no ssh key: runpodctl's key is absent and {DEFAULT_SSH_KEY} does not exist")
        key = str(DEFAULT_SSH_KEY)
    return {"user": user, "host": host, "port": str(port), "key": key}


class Pod:
    def __init__(self, pod: dict):
        self.id = pod["id"]
        self.target = ssh_target(self.id)

    def _ssh_base(self) -> list[str]:
        t = self.target
        base = ["ssh", *SSH_OPTS, "-p", t["port"]]
        if t.get("key"):
            base += ["-i", str(Path(t["key"]).expanduser())]
        return base + [f"{t['user']}@{t['host']}"]

    def run(self, command: str, timeout: float = 3600, stream: bool = False) -> str:
        """Run one shell command non-interactively. `stream` inherits stdout so progress shows live."""
        args = self._ssh_base() + [command]
        if stream:
            proc = subprocess.run(args, timeout=timeout, text=True)
            out = ""
        else:
            proc = subprocess.run(args, timeout=timeout, text=True, capture_output=True)
            out = proc.stdout
            if proc.stderr.strip():
                print(proc.stderr.strip())
        if proc.returncode != 0:
            raise RunpodError(f"remote command failed ({proc.returncode}): {command[:120]}...\n{out[-2000:]}")
        return out

    def upload(self, root: Path, rel_paths: list[str], remote_dir: str) -> None:
        """tar over ssh: incremental enough for a few MB of code and parquet, no rsync needed on the pod."""
        tar = subprocess.Popen(
            ["tar", "czf", "-", "--exclude", "__pycache__", "-C", str(root), *rel_paths], stdout=subprocess.PIPE
        )
        proc = subprocess.run(
            self._ssh_base() + [f"mkdir -p {remote_dir} && tar xzf - -C {remote_dir}"],
            stdin=tar.stdout,
            capture_output=True,
            text=True,
            timeout=600,
        )
        tar.wait()
        if proc.returncode != 0 or tar.returncode != 0:
            raise RunpodError(f"upload failed: {proc.stderr}")

    def write_file(self, remote_path: str, content: str) -> None:
        """Send file content over stdin; nothing of it appears in a command line or log."""
        proc = subprocess.run(
            self._ssh_base() + [f"cat > {remote_path} && chmod 600 {remote_path}"],
            input=content,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RunpodError(f"writing {remote_path} failed: {proc.stderr}")

    def download(self, remote_path: str, local_path: Path) -> None:
        t = self.target
        tmp = local_path.with_name(local_path.name + ".tmp")
        args = ["scp", *SSH_OPTS, "-P", t["port"]]
        if t.get("key"):
            args += ["-i", str(Path(t["key"]).expanduser())]
        proc = subprocess.run(
            args + [f"{t['user']}@{t['host']}:{remote_path}", str(tmp)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RunpodError(f"download of {remote_path} failed: {proc.stderr}")
        tmp.rename(local_path)


def setup_pod(pod: Pod, extra_packages: list[str] = ()) -> None:
    pkgs = " ".join(f"'{p}'" for p in [*config.RUNPOD_PIP_PACKAGES, *extra_packages])
    script = f"""set -e
mkdir -p {config.RUNPOD_HF_HOME} {config.RUNPOD_REMOTE_DIR}
pip install --break-system-packages -q {pkgs}
python - <<'PY'
import torch, sentence_transformers, transformers
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), gpu)
print("sentence-transformers", sentence_transformers.__version__, "| transformers", transformers.__version__)
PY
"""
    print(pod.run(script, timeout=1200).strip())


def run_on_pod(
    *,
    uploads: list[str],
    command: str,
    downloads: list[tuple[str, Path]],
    env: dict[str, str] | None = None,
    extra_packages: list[str] = (),
    keep_pod: bool = False,
) -> None:
    """Provision (or reuse) the pod, sync inputs, run one stage command, pull outputs, tear down.

    `uploads` are paths relative to the repo root; `downloads` pair a remote path with a local destination.
    `env` is written to a `.env` in the remote project dir (config.py loads it), so secrets never appear
    on a command line or in a log.
    """
    t0 = time.time()
    existing = find_pod()
    pod_json = existing or create_pod()
    pod_id = pod_json["id"]
    origin = "reused" if existing else "created"
    try:
        pod = Pod(pod_json)
        t = pod.target
        where = f" on {pod_json.get('_gpu_id')} in {pod_json.get('_data_center')}" if not existing else ""
        print(f"pod {pod.id} {origin}{where} after {time.time() - t0:.0f}s: ssh {t['user']}@{t['host']}:{t['port']}")
        setup_pod(pod, extra_packages)
        pod.upload(config.ROOT, uploads, config.RUNPOD_REMOTE_DIR)
        print(f"uploaded {', '.join(uploads)}", flush=True)
        if env:
            pod.write_file(f"{config.RUNPOD_REMOTE_DIR}/.env", "".join(f"{k}={v}\n" for k, v in env.items()))
        pod.run(command, timeout=3 * 3600, stream=True)
        for remote_path, local_path in downloads:
            pod.download(remote_path, local_path)
            print(f"pulled {local_path}")
    finally:
        if keep_pod:
            print(f"pod {pod_id} left running (billing continues): delete with `runpodctl pod delete {pod_id}`")
        else:
            runpodctl("pod", "delete", pod_id)
            print(f"pod {pod_id} deleted ({origin} this run; {(time.time() - t0) / 60:.1f} min total)")


def _remote(rel: str) -> str:
    return f"{config.RUNPOD_REMOTE_DIR}/{rel}"


def _rel(path: Path) -> str:
    return str(path.relative_to(config.ROOT))


def run_embed_on_runpod(key: str, limit: int | None = None, keep_pod: bool = False) -> None:
    files = config.keyed_files(key)
    spec = config.EMBED_MODELS[key]
    print(f"runpod: embed with {spec['model']} on {config.RUNPOD_GPU_ID}; minutes of a sub-$1/hour pod", flush=True)
    command = (
        f"cd {config.RUNPOD_REMOTE_DIR} && env HF_HOME={config.RUNPOD_HF_HOME} PYTHONUNBUFFERED=1 "
        f"python pipeline/02_embed.py --model {key} --device cuda" + (f" --limit {limit}" if limit else "")
    )
    downloads = [] if limit else [(_remote(_rel(f)), f) for f in (files["npz"], files["meta"])]
    uploads = ["pipeline", _rel(config.PATHS["corpus"])]
    run_on_pod(uploads=uploads, command=command, downloads=downloads, keep_pod=keep_pod)
    if not limit:
        data = np.load(files["npz"], allow_pickle=True)
        emb = data["embeddings"]
        assert np.isfinite(emb).all() and len(data["cid"]) == emb.shape[0]
        print(f"verified {files['npz']} {emb.shape}")


def run_label_on_runpod(key: str, sweep: bool = False, keep_pod: bool = False) -> None:
    """Stage 04 on the pod: the keyphrase embedding is the same GPU workload as stage 02, and the LLM calls
    go out from the pod. ANTHROPIC_API_KEY travels in the remote .env, not on a command line."""
    files = config.keyed_files(key)
    if not config.ANTHROPIC_API_KEY and not sweep:
        raise RunpodError("ANTHROPIC_API_KEY is not set locally; nothing to hand the pod")
    print(f"runpod: label {key} on {config.RUNPOD_GPU_ID}{' (sweep only, no LLM)' if sweep else ''}", flush=True)
    command = (
        f"cd {config.RUNPOD_REMOTE_DIR} && env HF_HOME={config.RUNPOD_HF_HOME} PYTHONUNBUFFERED=1 "
        f"TOKENIZERS_PARALLELISM=false python pipeline/04_label_topics.py --embedding {key} --device cuda"
        + (" --sweep" if sweep else "")
    )
    outputs = [files[k] for k in ("labels", "topic_names", "cluster_tree", "labels_meta")]
    run_on_pod(
        uploads=["pipeline", _rel(config.PATHS["corpus"]), _rel(files["npz"]), _rel(files["umap"])],
        command=command,
        downloads=[] if sweep else [(_remote(_rel(f)), f) for f in outputs],
        env=None if sweep else {"ANTHROPIC_API_KEY": config.ANTHROPIC_API_KEY},
        extra_packages=config.RUNPOD_LABEL_PACKAGES,
        keep_pod=keep_pod,
    )
