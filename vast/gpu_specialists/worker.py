"""Configure the pinned, unmodified gateway workers for this installation."""
import dataclasses
import json
import os
import sys
from pathlib import Path


def configure(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    source = root / "source"
    import_paths = [str(source), str(source / "deploy/ocr"),
                    str(source / "evals/translation_ocr/real_quality")]
    sys.path[:0] = import_paths
    os.environ.update(config["environment"])
    # The accepted OCR supervisor executes a fresh Python interpreter for each
    # request. sys.path edits alone do not reach that disposable child.
    os.environ["PYTHONPATH"] = os.pathsep.join(import_paths +
        ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.environ["PATH"] = str(root / "bin") + ":" + os.environ["PATH"]
    return config


def main():
    import uvicorn
    config = configure(sys.argv[1])
    name = sys.argv[2]
    if name == "mt":
        from evals.multilang import gpu_specialist_server as gpu
        from deploy.gpu_stage import mt_on_demand
        artifacts = config["mt_artifacts"]
        gpu.MODEL = Path(artifacts["model"])
        gpu.SERVER = Path(artifacts["server"])
        gpu.COMPAT_LIBRARY = Path(artifacts["compat"])
        gpu.MODEL_SHA256 = artifacts["model_sha256"]
        gpu.SERVER_SHA256 = artifacts["server_sha256"]
        gpu.COMPAT_SHA256 = artifacts["compat_sha256"]
        gpu.RUNTIME_SO_HASHES = artifacts["libraries"]
        gpu.GPU_UUID = config["gpu"]["uuid"]
        gpu.EXPECTED_INCUMBENTS = frozenset(config["gpu"]["incumbents"])
        gpu.LOCK = Path(config["environment"]["OCR_SHARED_GPU_LOCK_PATH"])
        command = mt_on_demand._command

        def scoped_command(token_file):
            pinned = command(token_file)
            return dataclasses.replace(pinned, environment={
                **pinned.environment,
                "CUDA_VISIBLE_DEVICES": gpu.GPU_UUID,
                "LD_LIBRARY_PATH": config["environment"]["LD_LIBRARY_PATH"],
                "PATH": os.environ["PATH"],
                "SECAI_SPECIALIST_GPU_UUID": gpu.GPU_UUID,
            })

        mt_on_demand._command = scoped_command
        app = mt_on_demand.create_mt_stage_app()
        port = config["mt_port"]
    elif name == "ocr":
        os.environ["OCR_SERVICE_TOKEN"] = Path(config["ocr_token_file"]).read_text().strip()
        from qwen_ocr_supervisor import app
        port = config["ocr_port"]
    else:
        raise SystemExit("unknown worker")
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, access_log=False,
                timeout_graceful_shutdown=125)


if __name__ == "__main__":
    main()
