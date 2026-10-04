import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / "vast/gpu_specialists"
spec = importlib.util.spec_from_file_location("specialist_bootstrap", HELPERS / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class SpecialistsTests(unittest.TestCase):
    def test_dry_run_cannot_call_vast_network_or_ssh(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("vastai", "curl", "ssh", "scp"):
                executable = root / name
                executable.write_text("#!/bin/sh\necho unexpected-paid-or-network-call >&2\nexit 77\n")
                executable.chmod(0o755)
            env = {**os.environ, "PATH": str(root) + ":" + os.environ["PATH"]}
            result = subprocess.run(["bash", str(REPO / "serving-up-cyber-f1.sh"),
                                     "--dry-run", "--yes", "--replace-old"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("unexpected-paid", result.stderr)
            self.assertIn("DuyTa/sec-embedding", result.stdout)
            self.assertIn("DuyTa/sec-rerank", result.stdout)
            self.assertIn("SSH/SCP pinned runtime bundle", result.stdout)
            # Check the actual expanded onstart length, not compressed sources.
            onstart = result.stdout.split("onstart (đã giải mã, để review):\n", 1)[1]
            onstart = onstart.split("After base readiness:", 1)[0]
            self.assertLessEqual(len(onstart.replace("    ", "").encode()), 16384)

    def test_dry_run_redacts_explicit_and_inherited_api_key(self):
        for arguments in ([], ["--api-key", "sk-explicit-secret-fixture"]):
            result = subprocess.run(["bash", str(REPO / "serving-up-cyber-f1.sh"), "--dry-run", *arguments],
                                    env={**os.environ, "API_KEY": "sk-inherited-secret-fixture"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("sk-explicit-secret-fixture", result.stdout)
            self.assertNotIn("sk-inherited-secret-fixture", result.stdout)
            self.assertIn("API_KEY=<redacted>", result.stdout)

    def test_legacy_three_service_dry_run(self):
        result = subprocess.run(["bash", str(REPO / "serving-up-cyber-f1.sh"), "--dry-run",
                                 "--disable-specialists", "--gpus", "2", "--llm-model", "custom/model"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("custom/model", result.stdout)
        self.assertNotIn("After base readiness:", result.stdout)

    def test_remote_installer_dry_run_is_read_only(self):
        result = subprocess.run(["bash", str(REPO / "serving-specialists-install.sh"),
                                 "--host", "root@192.165.134.28", "--port", "10963", "--dry-run",
                                 "--reuse-artifacts", "--mt-token-file", "/private/token"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--reuse-artifacts", result.stdout)

    def test_exact_bundle_extract_and_tamper_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap.extract_source(root)
            manifest = json.loads((root / "runtime-manifest.json").read_text())
            self.assertEqual(manifest["gateway_revision"], "0650f9d7d0b40f9eefb94f3059e1516d842ac112")
            path = root / "source/deploy/gpu_stage/mt_on_demand.py"
            bootstrap.check_hash(path, manifest["files"]["deploy/gpu_stage/mt_on_demand.py"])
            path.write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                bootstrap.attest({"root": str(root), "enabled": {"ocr": False, "mt": False}})

    def test_extract_when_bundle_already_uploaded_to_root(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("runtime-manifest.json", "runtime.tar.gz"):
                shutil.copyfile(HELPERS / name, root / name)
            with patch.object(bootstrap, "HERE", root):
                bootstrap.extract_source(root)

    def test_private_token_created_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = Path(bootstrap.token_file(root, None, "mt"))
            original = path.read_bytes()
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            self.assertEqual(bootstrap.token_file(root, str(path), "mt"), str(path))
            self.assertEqual(path.read_bytes(), original)
            path.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "private token"):
                bootstrap.token_file(root, str(path), "mt")

    def test_venv_python_symlink_is_not_resolved_to_base_interpreter(self):
        import argparse
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / "venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            args = argparse.Namespace(python=str(python), reuse_artifacts=True, disable_ocr=True)
            with patch.object(bootstrap, "run") as mocked:
                selected = bootstrap.prepare_python(args, root)
            self.assertEqual(selected, str(python))
            self.assertEqual(mocked.call_args.args[0], python)

    def test_reuse_missing_python_fails_without_creating_venv(self):
        import argparse
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for selected in (str(root / "missing-python"), None):
                args = argparse.Namespace(python=selected, reuse_artifacts=True, disable_ocr=True)
                with patch.object(bootstrap, "run") as mocked:
                    with self.assertRaisesRegex(RuntimeError, "existing"):
                        bootstrap.prepare_python(args, root)
                mocked.assert_not_called()
                self.assertFalse((root / "venv").exists())

    def test_disposable_ocr_interpreter_inherits_accounting_helper_import(self):
        worker_spec = importlib.util.spec_from_file_location("specialist_worker", HELPERS / "worker.py")
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helper = root / "source/deploy/ocr/server.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("ACCOUNTING_HELPER='selected-private-ocr-helper'\n")
            child = root / "source/evals/translation_ocr/real_quality/child.py"
            child.parent.mkdir(parents=True)
            child.write_text("import server\nprint(server.ACCOUNTING_HELPER)\nprint(server.__file__)\n")
            (root / "config.json").write_text(json.dumps({"environment": {}}))
            # Execute the same fresh-interpreter boundary as the OCR supervisor.
            with patch.dict(os.environ, {"PYTHONPATH": ""}), patch.object(sys, "path", list(sys.path)):
                worker.configure(root)
                result = subprocess.run([sys.executable, str(child)], env=dict(os.environ),
                                        cwd=child.parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["selected-private-ocr-helper", str(helper)])

    def test_ready_checks_auth_routes_then_serial_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"enabled": {"mt": True, "ocr": True}, "python": sys.executable}
            (root / "config.json").write_text(json.dumps(config))
            responses = [(401, {}), (200, {"translated_text": "test"}), (200, {"translated_text": "Xin chào."}),
                         (401, {}), (422, {}), (200, {"blocks": [{"text": "SECAI 2026"}]})]
            with patch.object(bootstrap, "attest"), patch.object(bootstrap, "run", return_value="png"), \
                    patch.object(bootstrap, "post", side_effect=responses) as posted:
                bootstrap.ready(root, smoke=True)
            self.assertEqual([call.args[1] for call in posted.call_args_list], ["mt", "mt", "mt", "ocr", "ocr", "ocr"])
            self.assertEqual(posted.call_args_list[2].args[2]["target_lang"], "vi")
            self.assertEqual(posted.call_args_list[-1].args[2]["model"], "Qwen/Qwen3-VL-2B-Instruct")

    def test_capture_scopes_two_gpu_host_to_selected_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap.extract_source(root)
            config = {"root": str(root), "environment": {"OCR_SHARED_GPU_LOCK_PATH": str(root / "gpu/admission.lock")}}
            calls = []
            def query(*args, **kwargs):
                calls.append(args)
                return "GPU-selected, 8772" if "--query-gpu=uuid,memory.free" in args else "101\n102\n103"
            with patch.object(bootstrap, "run", query), patch.object(bootstrap.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                bootstrap.capture_gpu(config, "1", 3)
            self.assertEqual(config["gpu"], {"uuid": "GPU-selected", "incumbents": [101, 102, 103]})
            self.assertEqual(calls[0][1:3], ("--id", "1"))
            self.assertEqual(calls[1][1:3], ("--id", "GPU-selected"))
            self.assertIn('--id "$SECAI_SPECIALIST_GPU_UUID"', (root / "bin/nvidia-smi").read_text())
            self.assertEqual(config["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-selected")

    def test_low_free_gpu_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap.extract_source(root)
            config = {"root": str(root), "environment": {"OCR_SHARED_GPU_LOCK_PATH": str(root / "admission.lock")}}
            with patch.object(bootstrap, "run", side_effect=["GPU-selected, 8191", "101\n102\n103"]):
                with self.assertRaisesRegex(RuntimeError, "8192"):
                    bootstrap.capture_gpu(config, "0", 3)

    def test_shared_cleanup_poison_cannot_be_recaptured(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap.extract_source(root)
            (root / "poisoned-cleanup.json").write_text("{}")
            config = {"root": str(root), "environment": {"OCR_SHARED_GPU_LOCK_PATH": str(root / "admission.lock")}}
            with self.assertRaises(Exception) as raised:
                bootstrap.capture_gpu(config, "0", 3)
            self.assertEqual(str(raised.exception), "POISONED")

    def test_worker_restart_preserves_other_worker(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "logs").mkdir()
            shutil.copyfile(HELPERS / "supervisor.py", root / "supervisor.py")
            (root / "config.json").write_text(json.dumps({"python": sys.executable, "enabled": {"mt": True, "ocr": True}}))
            (root / "worker.py").write_text(
                "import os,sys,time\nfrom pathlib import Path\n"
                "with (Path(sys.argv[1])/'events').open('a') as f: f.write(sys.argv[2]+' '+str(os.getpid())+'\\n')\n"
                "while True: time.sleep(.1)\n")
            supervisor = subprocess.Popen([sys.executable, str(root / "supervisor.py"), str(root)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            def events(count):
                until = time.monotonic() + 12
                while time.monotonic() < until:
                    rows = (root / "events").read_text().splitlines() if (root / "events").exists() else []
                    if len(rows) >= count:
                        return dict(row.split() for row in rows), rows
                    time.sleep(.1)
                self.fail("worker startup/restart timeout")
            try:
                initial, _ = events(2)
                os.kill(int(initial["mt"]), signal.SIGTERM)
                restarted, rows = events(3)
                self.assertNotEqual(restarted["mt"], initial["mt"])
                self.assertEqual(restarted["ocr"], initial["ocr"])
                self.assertEqual(sum(row.startswith("ocr ") for row in rows), 1)
                bootstrap.stop(root)
                self.assertEqual(supervisor.wait(timeout=3), 0)
                self.assertFalse(bootstrap.is_running(root))
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    supervisor.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
