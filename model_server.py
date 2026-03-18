#!/usr/bin/env python3
"""
Persistent model server — loads pi05 once, serves get_action over Unix socket.
Client: experiments/robot/model_client.py  |  Flag: --use_model_server
Protocol: 4-byte big-endian length prefix + pickle payload.
"""

import argparse
import os
import pickle
import socket
import struct
import sys

SOCKET_PATH = "/tmp/vla_model.sock"


def _send(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(data)) + data)


def _recv(conn):
    raw = _recvn(conn, 4)
    if not raw:
        return None
    n = struct.unpack(">I", raw)[0]
    return pickle.loads(_recvn(conn, n))


def _recvn(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def serve(cfg, model, action_head, proprio_projector, noisy_action_projector, processor):
    import numpy as np
    from experiments.robot.robot_utils import get_action

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    print(f"[model_server] Listening on {SOCKET_PATH}")
    print("[model_server] Ready. Ctrl+C to stop.\n")

    try:
        while True:
            conn, _ = server.accept()
            try:
                req = _recv(conn)
                if req is None:
                    continue
                obs = req["obs"]
                task_label = req["task_label"]
                # Merge any per-request cfg overrides (e.g. unnorm_key)
                for k, v in req.get("cfg_overrides", {}).items():
                    setattr(cfg, k, v)

                actions = get_action(
                    cfg, model, obs, task_label,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=getattr(cfg, "use_film", False),
                )
                _send(conn, {"actions": actions})
            except Exception as e:
                import traceback
                _send(conn, {"error": str(e), "tb": traceback.format_exc()})
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[model_server] Shutting down.")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--model_family", default="pi05")
    parser.add_argument("--socket_path", default=SOCKET_PATH)
    args = parser.parse_args()

    SOCKET_PATH = args.socket_path

    # Bootstrap cfg from GenerateConfig defaults
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LIBERO"))
    from experiments.robot.libero.vla_eval import GenerateConfig, initialize_model
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        model_family=args.model_family,
        task_suite_name="libero_90",  # needed for unnorm key validation
    )
    set_seed_everywhere(7)

    print(f"[model_server] Loading {args.model_family} from {args.pretrained_checkpoint} ...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    print("[model_server] Model loaded.\n")

    serve(cfg, model, action_head, proprio_projector, noisy_action_projector, processor)
