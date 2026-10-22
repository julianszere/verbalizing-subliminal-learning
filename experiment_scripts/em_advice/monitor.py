"""Locally mirror the remote sweep and stop its GPU on completion or timeout.

Uses the existing local Vast.ai CLI credentials; never uploads credentials.
Run in a persistent local process, with stdout/stderr redirected to a log.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import shlex


def complete_result(record, checkpoint_exists, expected_train, skip_held_out):
    provenance = record.get("provenance", {})
    if skip_held_out:
        evaluation_ok = (provenance.get("held_out_evaluated") is False
                         and "hypothesis_test_nll" not in record
                         and not any(k == "test" or k.endswith("_test") for k in record.get("losses", {})))
    else:
        evaluation_ok = "hypothesis_test_nll" in record
    return (provenance.get("model") == "Qwen/Qwen2.5-14B-Instruct"
            and provenance.get("recipe") == "main"
            and provenance.get("train") == expected_train
            and provenance.get("test") == 500 and evaluation_ok and checkpoint_exists)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=14.5)
    parser.add_argument("--min-credit", type=float, help="Stop with artifacts preserved below this balance")
    parser.add_argument("--train-samples", type=int, help="Expected fixed training count per domain")
    parser.add_argument("--skip-held-out", action="store_true")
    parser.add_argument("--remote-out", default="/workspace/salve/outputs/em_advice")
    parser.add_argument("--remote-log", default="/workspace/salve/setup.log")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", args.port,
           f"root@{args.host}"]
    status = {"instance": args.instance, "state": "monitoring", "max_hours": args.hours}
    (args.out / "monitor.json").write_text(json.dumps(status, indent=2))
    try:
        while time.monotonic() < deadline:
            if args.min_credit is not None:
                account = subprocess.run(["vastai", "show", "user", "--raw"],
                                         capture_output=True, text=True, timeout=40)
                if account.returncode == 0:
                    credit = json.loads(account.stdout).get("credit")
                    status["credit"] = credit
                    if credit is not None and credit < args.min_credit:
                        status["state"] = "low_credit"
                        break
            result = subprocess.run(ssh + [f"cat {shlex.quote(args.remote_out + '/sweep.exit')} 2>/dev/null"],
                                    capture_output=True, text=True, timeout=40)
            # SCP writes into a separate mirror; never mixes with older benchmark runs.
            copy = subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-P", args.port, "-r", f"root@{args.host}:{args.remote_out}",
                str(args.out)], capture_output=True, text=True, timeout=180)
            subprocess.run(["scp", "-o", "BatchMode=yes", "-P", args.port,
                f"root@{args.host}:{args.remote_log}", str(args.out / "sweep.log")],
                capture_output=True, timeout=45)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                status["remote_exit"] = int(result.stdout.strip())
                status["copy_ok"] = copy.returncode == 0
                complete = []
                for domain in ("medical", "finance", "sports"):
                    path = args.out / Path(args.remote_out).name / domain / "result.json"
                    if path.exists():
                        record = json.loads(path.read_text(encoding="utf-8"))
                        expected_train = args.train_samples or (6549 if domain == "medical" else 5500)
                        if complete_result(record, path.with_name("soft_prompt.pt").exists(),
                                           expected_train, args.skip_held_out):
                            complete.append(domain)
                status["completed_domains"] = complete
                status["state"] = "complete" if len(complete) == 3 and status["remote_exit"] == 0 and status["copy_ok"] else "incomplete"
                break
            status["last_check_unix"] = time.time()
            status["last_copy_ok"] = copy.returncode == 0
            (args.out / "monitor.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            time.sleep(60)
        else:
            status["state"] = "time_limit"
    except Exception as error:
        status.update(state="monitor_error", error=str(error))
    finally:
        # Leave the disk available for investigating or resuming an incomplete run.
        stopped = subprocess.run(["vastai", "stop", "instance", args.instance, "--raw"],
                                 capture_output=True, text=True, timeout=90)
        status["stop_returncode"] = stopped.returncode
        (args.out / "monitor.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
