"""Guard data separation and automatic GPU shutdown without requiring CUDA."""
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def temporary_directory():
    root = Path(__file__).resolve().parents[1] / "outputs" / "em_advice_tests"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "experiment_scripts" / "em_advice" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"em_advice_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdviceDataTests(unittest.TestCase):
    def test_grouped_split_preserves_rows_and_prevents_question_leakage(self):
        driver = load_script("run")
        rows = [{"messages": [{"role": "user", "content": f"question {i // 2}"},
                               {"role": "assistant", "content": f"response {i}"}]}
                for i in range(600)]
        with temporary_directory() as directory:
            root = Path(directory)
            raw = "\n".join(json.dumps(r) for r in rows).encode()
            (root / driver.FILES["finance"]).write_bytes(raw)
            with patch.dict(driver.COUNTS, finance=600), patch.dict(
                    driver.HASHES, finance=hashlib.sha256(raw).hexdigest()):
                train, test, provenance = driver.prepare(root, "finance", 42)
                again = driver.prepare(root, "finance", 42)
            self.assertEqual((len(train), len(test)), (100, 500))
            self.assertEqual((train, test, provenance), again)
            self.assertFalse({r["messages"][0]["content"] for r in train}
                             & {r["messages"][0]["content"] for r in test})
            self.assertEqual(sorted(r["source_index"] for r in train + test), list(range(600)))

    def test_changed_source_is_rejected(self):
        driver = load_script("run")
        with temporary_directory() as directory:
            root = Path(directory)
            (root / driver.FILES["medical"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                driver.prepare(root, "medical", 42)


class AdviceMonitorTests(unittest.TestCase):
    def test_complete_requires_artifacts_and_stops_gpu(self):
        monitor = load_script("monitor")
        with temporary_directory() as directory:
            out = Path(directory)
            for domain, count in (("medical", 6549), ("finance", 5500), ("sports", 5500)):
                folder = out / "em_advice" / domain
                folder.mkdir(parents=True)
                (folder / "soft_prompt.pt").write_bytes(b"test checkpoint")
                (folder / "result.json").write_text(json.dumps({"provenance": {
                    "model": "Qwen/Qwen2.5-14B-Instruct", "recipe": "main", "train": count,
                    "test": 500}, "hypothesis_test_nll": {}}))
            calls = []
            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "0\n", "")
            with patch("sys.argv", ["monitor", "--instance", "123", "--host", "host",
                 "--port", "22", "--out", str(out)]), patch.object(
                    monitor.subprocess, "run", side_effect=fake_run):
                monitor.main()
            status = json.loads((out / "monitor.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(calls[-1], ["vastai", "stop", "instance", "123", "--raw"])

    def test_time_limit_stops_gpu(self):
        monitor = load_script("monitor")
        with temporary_directory() as directory:
            with patch("sys.argv", ["monitor", "--instance", "123", "--host", "host",
                 "--port", "22", "--out", directory, "--hours", "0"]), patch.object(
                    monitor.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                monitor.main()
            self.assertEqual(json.loads((Path(directory) / "monitor.json").read_text())["state"], "time_limit")
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
