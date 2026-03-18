"""
model_client.py

Drop-in replacement for get_action that forwards requests to model_server.py
over a Unix socket instead of running inference locally.

Usage in eval script:
    from experiments.robot.model_client import get_action_remote as get_action
    # or set --use_model_server flag (handled in vla_eval_failures.py)
"""

import pickle
import socket
import struct

SOCKET_PATH = "/tmp/vla_model.sock"


def _send(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv(sock):
    raw = _recvn(sock, 4)
    if not raw:
        raise ConnectionError("Server closed connection")
    n = struct.unpack(">I", raw)[0]
    return pickle.loads(_recvn(sock, n))


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def get_action_remote(cfg, model, obs, task_label,
                      processor=None, action_head=None,
                      proprio_projector=None, noisy_action_projector=None,
                      use_film=False):
    """
    Same signature as robot_utils.get_action — sends request to model_server.
    `model` is ignored (None is fine); inference happens on the server.
    """
    cfg_overrides = {}
    if hasattr(cfg, "unnorm_key") and cfg.unnorm_key:
        cfg_overrides["unnorm_key"] = cfg.unnorm_key

    req = {"obs": obs, "task_label": task_label, "cfg_overrides": cfg_overrides}

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(SOCKET_PATH)
        _send(sock, req)
        resp = _recv(sock)
    finally:
        sock.close()

    if "error" in resp:
        raise RuntimeError(f"[model_server] {resp['error']}\n{resp.get('tb','')}")

    return resp["actions"]
