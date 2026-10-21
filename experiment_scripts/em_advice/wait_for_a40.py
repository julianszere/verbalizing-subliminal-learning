"""Wait for an affordable on-demand A40, then start the 100-row experiment."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def choose_offer(offers, max_hourly):
    offers = [offer for offer in offers if offer.get("num_gpus") == 1
              and offer.get("gpu_name") == "A40" and offer.get("rentable")
              and offer.get("reliability", 0) >= 0.95
              and offer.get("dph_total", float("inf")) <= max_hourly]
    return min(offers, key=lambda offer: offer["dph_total"], default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-minutes", type=float, default=60)
    parser.add_argument("--max-hourly", type=float, default=0.65)
    parser.add_argument("--out", type=Path, default=Path("outputs/em_advice_100_remote"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    state = {"state": "waiting_for_a40", "train_samples_per_domain": 100,
             "max_hourly": args.max_hourly, "wait_minutes": args.wait_minutes}

    def save(**updates):
        state.update(updates)
        (args.out / "queue_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def run(command, timeout=60, check=True):
        return subprocess.run(command, cwd=root, capture_output=True, text=True,
                              timeout=timeout, check=check)

    save()
    deadline = time.monotonic() + args.wait_minutes * 60
    instance = None
    try:
        while time.monotonic() < deadline:
            offers = json.loads(run(["vastai", "search", "offers",
                "gpu_name=A40 num_gpus=1 rentable=true", "--storage", "80",
                "--order", "dph_total", "--limit", "20", "--raw"]).stdout)
            offer = choose_offer(offers, args.max_hourly)
            if offer is not None:
                result = json.loads(run(["vastai", "create", "instance", str(offer["id"]),
                    "--image", "vastai/pytorch:@vastai-automatic-tag", "--disk", "80",
                    "--ssh", "--direct", "--cancel-unavail", "--label", "salve-em-advice-100-a40",
                    "--raw"]).stdout)
                if result.get("success") and result.get("new_contract"):
                    instance = str(result["new_contract"])
                    save(state="provisioning", instance=instance, hourly=offer["dph_total"])
                    break
            save(last_search_unix=time.time())
            time.sleep(60)
        else:
            save(state="capacity_unavailable")
            return

        boot_deadline = time.monotonic() + 600
        while time.monotonic() < boot_deadline:
            item = json.loads(run(["vastai", "show", "instance", instance, "--raw"]).stdout)
            if item.get("actual_status") in ("exited", "offline", "unknown"):
                raise RuntimeError("A40 failed to provision")
            if item.get("actual_status") == "running":
                host, port = item["ssh_host"], str(item["ssh_port"])
                ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                       "-o", "ConnectTimeout=15", "-p", port, f"root@{host}"]
                ready = run(ssh + ["mkdir -p /workspace/salve/data/em_advice"], check=False)
                if ready.returncode == 0:
                    break
            time.sleep(15)
        else:
            raise RuntimeError("A40 provisioning timed out")

        scp = ["scp", "-o", "BatchMode=yes", "-P", port]
        run(scp + ["data/em_advice/source.tar.gz", "experiment_scripts/em_advice/setup.sh",
                    f"root@{host}:/workspace/salve/"], timeout=180)
        run(scp + [f"data/em_advice/{name}" for name in
                    ("bad_medical_advice.jsonl", "risky_financial_advice.jsonl", "extreme_sports.jsonl")]
                    + [f"root@{host}:/workspace/salve/data/em_advice/"], timeout=180)
        run(ssh + ["cd /workspace/salve; nohup bash setup.sh > setup.log 2>&1 < /dev/null & echo $!"])
        with (args.out / "monitor.log").open("w") as log, (args.out / "monitor.err").open("w") as err:
            monitor = subprocess.Popen([sys.executable, "-u", "experiment_scripts/em_advice/monitor.py",
                "--instance", instance, "--host", host, "--port", port, "--out", str(args.out),
                "--train-samples", "100", "--min-credit", "0.35"], cwd=root,
                stdout=log, stderr=err, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        save(state="launched", host=host, port=port, monitor_pid=monitor.pid)
    except Exception as error:
        save(state="launch_failed", error=str(error))
        if instance is not None:
            run(["vastai", "stop", "instance", instance, "--raw"], check=False)


if __name__ == "__main__":
    main()
