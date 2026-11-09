import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def dry_run(script, **environment):
    return subprocess.run(
        ["bash", str(ROOT / "blackwell" / script), "--dry-run"],
        env={**os.environ, "API_KEY": "fixture-private-key", **environment},
        capture_output=True, text=True,
    )


class BlackwellProfileTests(unittest.TestCase):
    def test_llm_profile_shards_one_model_across_four_gpus(self):
        result = dry_run("llm.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3", result.stdout)
        self.assertIn("--tensor-parallel-size 4", result.stdout)
        self.assertIn("--enable-expert-parallel", result.stdout)
        self.assertIn('cpu_offload\\":false', result.stdout)
        self.assertIn("--max-model-len 131072", result.stdout)
        self.assertIn("--tool-call-parser qwen3_coder", result.stdout)
        self.assertNotIn("fixture-private-key", result.stdout + result.stderr)

    def test_llm_routing_profiles_are_explicit(self):
        result = dry_run("llm.sh", PARALLEL_MODE="tp4", PLE_LOCATION="cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--enable-expert-parallel", result.stdout)
        self.assertIn('cpu_offload\\":true', result.stdout)
        invalid = dry_run("llm.sh", PARALLEL_MODE="tp3")
        self.assertNotEqual(invalid.returncode, 0)

    def test_llm_two_replica_profiles_use_pcie_pairs(self):
        first = dry_run("llm.sh", PARALLEL_MODE="tp2_ep2", LLM_GPU_IDS="0,1", LLM_PORT="8001")
        second = dry_run("llm.sh", PARALLEL_MODE="tp2_ep2", LLM_GPU_IDS="2,3", LLM_PORT="8003")
        for result, ids, port in ((first, "0,1", "8001"), (second, "2,3", "8003")):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={ids}", result.stdout)
            self.assertIn("--tensor-parallel-size 2", result.stdout)
            self.assertIn("--enable-expert-parallel", result.stdout)
            self.assertIn(f"--port {port}", result.stdout)
        invalid = dry_run("llm.sh", PARALLEL_MODE="tp2", LLM_GPU_IDS="0,2")
        self.assertNotEqual(invalid.returncode, 0)

    def test_llm_tuning_flags_are_explicit(self):
        result = dry_run("llm.sh", LLM_MAX_BATCHED_TOKENS="32768",
                         LLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}',
                         LLM_COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8]}',
                         LLM_KERNEL_CONFIG='{"linear_backend":"flashinfer_b12x"}',
                         LLM_ASYNC_SCHEDULING="1",
                         LLM_DISABLE_CUSTOM_ALL_REDUCE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-num-batched-tokens 32768", result.stdout)
        self.assertIn("--speculative-config", result.stdout)
        self.assertIn("--compilation-config", result.stdout)
        self.assertIn("--kernel-config", result.stdout)
        self.assertIn("flashinfer_b12x", result.stdout)
        self.assertIn("--async-scheduling", result.stdout)
        self.assertIn("--disable-custom-all-reduce", result.stdout)

    def test_5090_profile_routes_auxiliary_services_only(self):
        result = dry_run("aux.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", result.stdout)
        self.assertIn("DuyTa/sec-embedding", result.stdout)
        self.assertIn("DuyTa/sec-rerank", result.stdout)
        embed, rerank, *_ = result.stdout.splitlines()
        self.assertIn("--max-model-len 1024", embed)
        self.assertIn("--max-model-len 2048", rerank)
        self.assertIn("ATH-MaaS/OvisOCR3", result.stdout)
        self.assertIn("--gdn-prefill-backend triton", result.stdout)
        self.assertIn("min_pixels", result.stdout)
        self.assertIn("max_pixels", result.stdout)
        self.assertIn("blackwell/ocr_adapter.py", result.stdout)
        self.assertNotIn("Hy-MT2", result.stdout)
        self.assertNotIn("Qwen3-VL-2B", result.stdout)
        self.assertNotIn("Qwen3.8-Flash-Next", result.stdout)
        self.assertNotIn("fixture-private-key", result.stdout + result.stderr)

    def test_ocr_can_be_disabled_without_changing_embedding_routes(self):
        result = dry_run("aux.sh", ENABLE_OCR="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ATH-MaaS/OvisOCR3", result.stdout)
        self.assertIn("DuyTa/sec-embedding", result.stdout)

    def test_benchmark_matrix_keeps_ple_comparison_fair(self):
        result = dry_run("bench.sh", PROFILE="tp4_ep4_ple_gpu", BASE_URL="http://127.0.0.1:8001")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("vllm bench serve"), 12)
        self.assertIn("--endpoint /v1/chat/completions", result.stdout)
        self.assertIn("--input-len 32768", result.stdout)
        self.assertIn("--max-concurrency 32", result.stdout)
        self.assertIn("--input-len 126976", result.stdout)
        self.assertNotIn("fixture-private-key", result.stdout + result.stderr)

    def test_benchmark_uses_served_name_and_local_tokenizer(self):
        result = dry_run("bench.sh", LLM_MODEL="/workspace/qwen38-orcarouter")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4", result.stdout)
        self.assertIn("--tokenizer /workspace/qwen38-orcarouter", result.stdout)

    def test_quick_benchmark_screen_covers_latency_and_context(self):
        result = dry_run("bench.sh", BENCH_MODE="quick")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("vllm bench serve"), 4)
        self.assertIn("--input-len 2048 --output-len 256", result.stdout)
        self.assertIn("--input-len 126976 --output-len 256", result.stdout)

    def test_routes_keep_llm_and_aux_instances_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            llm = Path(folder) / "llm.json"
            aux = Path(folder) / "aux.json"
            llm.write_text(json.dumps({"public_ipaddr": "192.0.2.10", "ports": {
                "8001/tcp": [{"HostPort": "40101"}]}}))
            aux.write_text(json.dumps({"public_ipaddr": "192.0.2.20", "ports": {
                "8000/tcp": [{"HostPort": "40200"}],
                "8002/tcp": [{"HostPort": "40202"}]}}))
            result = subprocess.run(["python3", str(ROOT / "blackwell/routes.py"),
                                     "--llm-json", str(llm), "--aux-json", str(aux)],
                                    capture_output=True, text=True,
                                    env={**os.environ, "API_KEY": "fixture-private-key"})
            self.assertEqual(result.returncode, 0, result.stderr)
            routes = json.loads(result.stdout)
            self.assertEqual(routes["llm"], "http://192.0.2.10:40101/v1")
            self.assertEqual(routes["embedding"], "http://192.0.2.20:40200")
            self.assertEqual(routes["rerank"], "http://192.0.2.20:40202")
            self.assertNotIn("fixture-private-key", result.stdout)

    def test_routes_refuse_missing_aux_port(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instance.json"
            path.write_text(json.dumps({"public_ipaddr": "192.0.2.10", "ports": {
                "8001/tcp": [{"HostPort": "40101"}]}}))
            result = subprocess.run(["python3", str(ROOT / "blackwell/routes.py"),
                                     "--llm-json", str(path), "--aux-json", str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("internal port 8000", result.stderr)

    def test_routes_support_two_replica_frontend_port(self):
        with tempfile.TemporaryDirectory() as folder:
            llm = Path(folder) / "llm.json"
            aux = Path(folder) / "aux.json"
            llm.write_text(json.dumps({"public_ipaddr": "192.0.2.10", "ports": {
                "8005/tcp": [{"HostPort": "40105"}]}}))
            aux.write_text(json.dumps({"public_ipaddr": "192.0.2.20", "ports": {
                "8000/tcp": [{"HostPort": "40200"}],
                "8002/tcp": [{"HostPort": "40202"}]}}))
            result = subprocess.run(["python3", str(ROOT / "blackwell/routes.py"),
                                     "--llm-json", str(llm), "--aux-json", str(aux),
                                     "--llm-port", "8005"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["llm"], "http://192.0.2.10:40105/v1")


if __name__ == "__main__":
    unittest.main()
