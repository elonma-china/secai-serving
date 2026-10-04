import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


class ReplicaProvisionTests(unittest.TestCase):
    def test_offline_five_model_plan_for_1_2_4_gpu(self):
        for count in (1, 2, 4):
            selected = count - 1
            result = subprocess.run(["bash", str(REPO / "serving-up-cyber-f1.sh"), "--dry-run",
                                     "--gpus", str(count), "--specialist-gpu", str(selected)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"NUM_GPUS={count}", result.stdout)
            self.assertIn(f"install OCR/MT on GPU {selected}", result.stdout)
            self.assertIn("--smoke", result.stdout)

    def test_invalid_gpu_selection_fails_before_vast_calls(self):
        for arguments in (["--gpus", "0"], ["--gpus", "5"], ["--gpus", "bad"],
                          ["--gpus", "1", "--specialist-gpu", "1"],
                          ["--gpus", "4", "--specialist-gpu", "-1"],
                          ["--gpus", "2", "--specialist-gpu", "bad"]):
            with tempfile.TemporaryDirectory() as folder:
                blocked = Path(folder) / "vastai"
                blocked.write_text("#!/bin/sh\necho unexpected-vast-call >&2\nexit 77\n")
                blocked.chmod(0o755)
                result = subprocess.run(["bash", str(REPO / "serving-up-cyber-f1.sh"), *arguments],
                                        env={**os.environ, "PATH": folder + ":" + os.environ["PATH"]},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("unexpected-vast-call", result.stderr)
                self.assertNotIn("API key", result.stderr)

    def run_mock_startup(self, count, actual=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            commands = root / "bin"
            commands.mkdir()
            trace = root / "calls.jsonl"
            python = commands / "python3"
            python.write_text(f"#!{sys.executable}\n" +
                "import os,sys,json\nargs=sys.argv[1:]\n"
                "if args[:2]==['-m','vllm.entrypoints.openai.api_server']:\n"
                " data={'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'port':args[args.index('--port')+1],'model':args[args.index('--model')+1]}\n"
                "elif args and args[0].endswith('resident_smoke.py'):\n data={'smoke':args}\n"
                "else:\n os.execv(sys.executable,[sys.executable,*args])\n"
                "with open(os.environ['TRACE'],'a') as f: f.write(json.dumps(data)+'\\n')\n")
            python.chmod(0o755)
            smi = commands / "nvidia-smi"
            smi.write_text("#!/bin/sh\ni=0\nwhile [ $i -lt \"$ACTUAL_GPUS\" ]; do echo \"GPU $i: NVIDIA A100-SXM4-80GB\"; i=$((i+1)); done\n")
            smi.chmod(0o755)
            for name in ("nginx", "envsubst", "curl"):
                executable = commands / name
                executable.write_text("#!/bin/sh\nexit 0\n")
                executable.chmod(0o755)
            env = {**os.environ, "PATH": str(commands) + ":" + os.environ["PATH"],
                   "TRACE": str(trace), "ACTUAL_GPUS": str(actual or count), "NUM_GPUS": str(count),
                   "API_KEY": "private-unit-fixture", "HF_HUB_ENABLE_HF_TRANSFER": "0",
                   "HF_HOME": str(root / "hf"), "LOG_MAIN": str(root / "onstart.log"),
                   "LOG_DIR": str(root / "logs"), "REPO_DIR": str(REPO),
                   "NGINX_CONF_DIR": str(root / "nginx"), "NGINX_DEFAULT_SITE": str(root / "default"),
                   "SECAI_SPECIALISTS_ENABLED": "0"}
            result = subprocess.run(["bash", str(REPO / "vast/on_start_cyber_f1.sh")],
                                    env=env, capture_output=True, text=True)
            until = time.monotonic() + 3
            while time.monotonic() < until:
                rows = [json.loads(row) for row in trace.read_text().splitlines()] if trace.exists() else []
                if result.returncode != 0 or len(rows) >= count * 4:
                    break
                time.sleep(.05)
            nginx = (root / "nginx/vllm.conf").read_text() if (root / "nginx/vllm.conf").exists() else ""
            return result, rows, nginx

    def test_startup_launches_three_residents_and_probes_each_gpu(self):
        for count in (1, 2, 4):
            result, rows, nginx = self.run_mock_startup(count)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            launches = [row for row in rows if "gpu" in row]
            self.assertEqual(len(launches), count * 3)
            for selected in range(count):
                self.assertEqual({row["model"] for row in launches if row["gpu"] == str(selected)},
                                 {"DuyTa/Cyber-F1-AWQ", "DuyTa/sec-embedding", "DuyTa/sec-rerank"})
                self.assertIn(f"127.0.0.1:{18101 + selected}", nginx)
                self.assertIn(f"127.0.0.1:{18001 + selected}", nginx)
                self.assertIn(f"127.0.0.1:{18201 + selected}", nginx)
            self.assertEqual(sum("smoke" in row for row in rows), count)

    def test_actual_gpu_count_mismatch_launches_nothing(self):
        result, rows, _ = self.run_mock_startup(4, actual=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(rows, [])
        self.assertIn("differs from actual", result.stdout)

    def test_resident_probes_use_actual_contract_routes_and_auth(self):
        spec = importlib.util.spec_from_file_location("resident_smoke", REPO / "vast/resident_smoke.py")
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        import io
        from contextlib import nullcontext
        bodies = [({"choices": [{"message": {"content": "OK"}}]}, "llm", "/v1/chat/completions"),
                  ({"data": [{"embedding": [0.1, 0.2]}]}, "embed", "/v1/embeddings"),
                  ({"results": [{"index": 0, "relevance_score": 0.9}]}, "rerank", "/rerank")]
        for payload, kind, route in bodies:
            with patch.object(smoke.urllib.request, "build_opener") as factory:
                factory.return_value.open.return_value = nullcontext(io.BytesIO(json.dumps(payload).encode()))
                smoke.probe("http://127.0.0.1:9999", route, "configured/model", {}, kind, "private-fixture")
            request = factory.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:9999" + route)
            self.assertEqual(request.get_header("Authorization"), "Bearer private-fixture")
            self.assertEqual(json.loads(request.data)["model"], "configured/model")


if __name__ == "__main__":
    unittest.main()
