"""Guard data separation and automatic GPU shutdown without requiring CUDA."""
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    def test_training_limit_keeps_held_out_split_and_exact_row_ids(self):
        driver = load_script("run")
        train = [{"source_index": i} for i in range(200)]
        test = [{"source_index": i} for i in range(200, 700)]
        original = {"medical": (train, test, {"train": 200})}
        limited = driver.limit_training(original, 100)
        selected, held_out, provenance = limited["medical"]
        self.assertEqual(len(selected), 100)
        self.assertIs(held_out, test)
        self.assertEqual(provenance["train_source_indices"], list(range(100)))
        self.assertEqual(provenance["train"], 100)
        self.assertEqual(original["medical"][2]["train"], 200)
        with self.assertRaises(ValueError):
            driver.limit_training(original, 201)

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
    def test_no_held_out_result_validation(self):
        monitor = load_script("monitor")
        record = {"provenance": {"model": "Qwen/Qwen2.5-14B-Instruct", "recipe": "main",
                  "train": 1000, "test": 500, "held_out_evaluated": False},
                  "losses": {"select": 1.0}}
        self.assertTrue(monitor.complete_result(record, True, 1000, True))
        self.assertFalse(monitor.complete_result(record, True, 100, True))
        record["losses"]["test"] = 2.0
        self.assertFalse(monitor.complete_result(record, True, 1000, True))

    def test_low_credit_stops_gpu_before_balance_runs_out(self):
        monitor = load_script("monitor")
        with temporary_directory() as directory:
            def fake_run(command, **kwargs):
                return subprocess.CompletedProcess(command, 0, json.dumps({"credit": 0.1}), "")
            with patch("sys.argv", ["monitor", "--instance", "123", "--host", "host",
                 "--port", "22", "--out", directory, "--min-credit", "0.35"]), patch.object(
                    monitor.subprocess, "run", side_effect=fake_run) as run:
                monitor.main()
            self.assertEqual(json.loads((Path(directory) / "monitor.json").read_text())["state"], "low_credit")
            self.assertEqual(run.call_args.args[0], ["vastai", "stop", "instance", "123", "--raw"])

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


class AdviceBatchingTests(unittest.TestCase):
    def test_a40_offer_selection_requires_capacity_reliability_and_price(self):
        launcher = load_script("wait_for_a40")
        base = {"gpu_name": "A40", "num_gpus": 1, "rentable": True, "reliability": .99}
        offers = [dict(base, id=1, dph_total=.8), dict(base, id=2, dph_total=.5),
                  dict(base, id=3, dph_total=.4, reliability=.7),
                  dict(base, id=4, dph_total=.3, gpu_name="A100 SXM4")]
        self.assertEqual(launcher.choose_offer(offers, .65)["id"], 2)
        self.assertIsNone(launcher.choose_offer(offers, .2))

    def test_fastest_batch_must_fit_memory_and_finish(self):
        batching = load_script("batching")
        records = [{"batch_size": 4, "ok": True, "seconds": 2., "worst_peak_bytes": 40},
                   {"batch_size": 8, "ok": True, "seconds": 1., "worst_peak_bytes": 70},
                   {"batch_size": 16, "ok": True, "seconds": .5, "worst_peak_bytes": 90},
                   {"batch_size": 32, "ok": False}]
        self.assertEqual(batching.fastest_safe(records, 80), 8)
        with self.assertRaises(RuntimeError):
            batching.fastest_safe(records, 30)


class AdviceSkipHeldOutTests(unittest.TestCase):
    def test_skip_never_tokenizes_or_scores_held_out_rows(self):
        driver = load_script("run")
        @dataclass
        class Soft:
            mini_batch_size: int = 2
            n_tokens: int = 128
            train_batch_size: int = 16
        @dataclass
        class Search:
            mini_batch_size: int = 4
            n_select: int = 256
            n_beams: int = 4
            branching: int = 16
            max_iters: int = 12
        class Objective:
            def __init__(self, *args):
                pass
            def prepare(self, rows):
                if any(user == "HELDOUT" for user, _ in rows):
                    raise AssertionError("Held-out rows reached tokenization")
                return rows
            def loss_of_text(self, *args, **kwargs):
                raise AssertionError("Unexpected post-search scoring")
            loss = loss_of_text
        row = lambda text, i: {"source_index": i, "messages": [
            {"role": "user", "content": text}, {"role": "assistant", "content": "answer"}]}
        train = [row("TRAIN", i) for i in range(1000)]
        held_out = [row("HELDOUT", 1000)]
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
        model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        z = SimpleNamespace(cpu=lambda: "checkpoint")
        salve_module = ModuleType("salve")
        objective_module = ModuleType("salve.objectives")
        objective_module.NLLObjective = Objective
        workflow = ModuleType("salve.salve")
        workflow.SoftPromptConfig, workflow.SearchConfig = Soft, Search
        workflow.load_model = lambda *args, **kwargs: (model, tokenizer)
        workflow.train_soft_prompt = lambda *args: (z, {})
        workflow.beam_search = lambda *args, **kwargs: ([{"score": 2.}], {"text": "Recovered", "score": 1.})
        torch_module = ModuleType("torch")
        torch_module.__version__ = "test"
        torch_module.cuda = SimpleNamespace(get_device_name=lambda: "fake GPU")
        torch_module.save = lambda *args: None
        with temporary_directory() as directory:
            modules = {"torch": torch_module, "salve": salve_module,
                       "salve.objectives": objective_module, "salve.salve": workflow}
            with patch.dict("sys.modules", modules), patch.object(driver, "prepare", return_value=(
                train, held_out, {"train": 1000, "test": 500})), patch("sys.argv", ["run",
                "--domains", "finance", "--out", directory, "--train-samples", "1000",
                "--search-samples", "100", "--bounded-search", "--skip-held-out"]):
                driver.main()
            result = json.loads((Path(directory) / "finance" / "result.json").read_text())
            self.assertEqual(set(result["losses"]), {"select", "empty_prompt_select"})
            self.assertNotIn("hypothesis_test_nll", result)
            self.assertFalse(result["provenance"]["held_out_evaluated"])
            self.assertEqual(result["provenance"]["search_config"]["n_select"], 100)


if __name__ == "__main__":
    unittest.main()
