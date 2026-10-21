"""Locally mirror the remote sweep and stop its GPU on completion or timeout.

Uses the existing local Vast.ai CLI credentials; never uploads credentials.
Run in a persistent local process, with stdout/stderr redirected to a log.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=14.5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", args.port,
           f"root@{args.host}"]
    status = {"instance": args.instance, "state": "monitoring", "max_hours": args.hours}
    (args.out / "monitor.json").write_text(json.dumps(status, indent=2))
    try:
        while time.monotonic() < deadline:
            result = subprocess.run(ssh + ["cat /workspace/salve/outputs/em_advice/sweep.exit 2>/dev/null"],
                                    capture_output=True, text=True, timeout=40)
            # SCP writes into a separate mirror; never mixes with older benchmark runs.
            copy = subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-P", args.port, "-r", f"root@{args.host}:/workspace/salve/outputs/em_advice",
                str(args.out)], capture_output=True, text=True, timeout=180)
            subprocess.run(["scp", "-o", "BatchMode=yes", "-P", args.port,
                f"root@{args.host}:/workspace/salve/setup.log", str(args.out / "sweep.log")],
                capture_output=True, timeout=45)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                status["remote_exit"] = int(result.stdout.strip())
                status["copy_ok"] = copy.returncode == 0
                complete = []
                for domain in ("medical", "finance", "sports"):
                    path = args.out / "em_advice" / domain / "result.json"
                    if path.exists():
                        record = json.loads(path.read_text(encoding="utf-8"))
                        provenance = record.get("provenance", {})
                        expected_train = 6549 if domain == "medical" else 5500
                        if (provenance.get("model") == "Qwen/Qwen2.5-14B-Instruct"
                            and provenance.get("recipe") == "main"
                            and provenance.get("train") == expected_train
                            and provenance.get("test") == 500
                            and "hypothesis_test_nll" in record
                            and path.with_name("soft_prompt.pt").exists()):
                            complete.append(domain)
                status["completed_domains"] = complete
                status["state"] = "complete" if len(complete) == 3 and status["remote_exit"] == 0 and status["copy_ok"] else "incomplete"
                break
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
