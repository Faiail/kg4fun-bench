#!/usr/bin/env python3
"""
Grid Search Memory Profiler for Cross-Encoding Experiments.
Runs a matrix of (task, model, batch_size, hidden_size) combinations in isolated
subprocesses, recording peak VRAM and OOM statuses to a CSV report in real-time.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Encoding VRAM Grid Search Profiler")
    parser.add_argument(
        "--tasks",
        type=str,
        default="node,edge,complete",
        help="Comma-separated list of tasks (default: node,edge,complete)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="bert,bge,modernbert,roberta",
        help="Comma-separated list of models (default: bert,bge,modernbert,roberta)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset1",
        help="Dataset name (default: dataset1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="hybrid",
        help="Split name (default: hybrid)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="512,256,128,64",
        help="Comma-separated list of batch sizes (default: 512,256,128,64)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="128,96",
        help="Comma-separated list of final hidden sizes (default: 128,96)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training batches per test (default: 5)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outs/memory_profile_grid.csv",
        help="Path for output CSV file (default: outs/memory_profile_grid.csv)",
    )
    return parser.parse_args()


def get_run_class(task: str) -> str:
    if task == "complete":
        return "SharedCrossTextCompleteRun"
    return "SharedCrossTextSingleItemRun"


def main():
    args = parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    hidden_sizes = [int(h.strip()) for h in args.hidden_sizes.split(",") if h.strip()]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_headers = [
        "timestamp",
        "task",
        "model",
        "batch_size",
        "hidden_size",
        "status",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "total_vram_gb",
        "vram_util_pct",
        "steps_completed",
        "steps_target",
        "duration_sec",
        "error_message",
    ]

    # Initialize CSV file with headers
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    total_runs = len(tasks) * len(models) * len(batch_sizes) * len(hidden_sizes)
    print("=" * 90)
    print("            CROSS-ENCODING MEMORY GRID SEARCH PROFILER")
    print("=" * 90)
    print(f" Tasks:        {tasks}")
    print(f" Models:       {models}")
    print(f" Batch Sizes:  {batch_sizes}")
    print(f" Hidden Dims:  {hidden_sizes}")
    print(f" Batches/test: {args.steps}")
    print(f" Dataset:      {args.dataset}/{args.split}")
    print(f" Total Tests:  {total_runs}")
    print(f" Output CSV:   {output_csv.resolve()}")
    print("=" * 90)

    results_all = []
    current_run = 0
    start_all = time.time()

    for task in tasks:
        run_cls = get_run_class(task)

        for model in models:
            config_path = f"configs/experiment/text/{task}/{args.dataset}/{args.split}/cross_{model}.yaml"
            if not os.path.exists(config_path):
                print(f"[Warning] Config {config_path} not found. Skipping {task}_{model}.")
                continue

            for hs in hidden_sizes:
                for bs in batch_sizes:
                    current_run += 1
                    test_id = f"[{current_run}/{total_runs}] {task.upper()} | {model} | BS={bs} | HS={hs}"
                    print(f"\n>>> Running {test_id} ...")

                    cmd = [
                        sys.executable,
                        "main.py",
                        "profile",
                        "--parameters",
                        config_path,
                        "--cls",
                        run_cls,
                        "--batch-size",
                        str(bs),
                        "--hidden-size",
                        str(hs),
                        "--steps",
                        str(args.steps),
                    ]

                    t0 = time.time()
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    duration = round(time.time() - t0, 2)
                    output_text = proc.stdout

                    # Extract JSON result line
                    result_data = None
                    for line in output_text.splitlines():
                        if line.startswith("PROFILE_RESULT:"):
                            try:
                                result_data = json.loads(line.replace("PROFILE_RESULT:", "").strip())
                            except Exception:
                                pass

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if result_data:
                        status = result_data.get("status", "UNKNOWN")
                        peak_alloc = result_data.get("peak_allocated_gb", 0.0)
                        peak_resv = result_data.get("peak_reserved_gb", 0.0)
                        total_vram = result_data.get("total_vram_gb", 0.0)
                        vram_pct = result_data.get("pct_used", 0.0)
                        steps_done = result_data.get("steps_completed", 0)
                        steps_tgt = result_data.get("steps_target", args.steps)
                        err_msg = result_data.get("error_message", "")
                    else:
                        status = "FAILED"
                        peak_alloc = 0.0
                        peak_resv = 0.0
                        total_vram = 0.0
                        vram_pct = 0.0
                        steps_done = 0
                        steps_tgt = args.steps
                        err_msg = "Process terminated without emitting PROFILE_RESULT"

                    row_dict = {
                        "timestamp": timestamp,
                        "task": task,
                        "model": model,
                        "batch_size": bs,
                        "hidden_size": hs,
                        "status": status,
                        "peak_allocated_gb": peak_alloc,
                        "peak_reserved_gb": peak_resv,
                        "total_vram_gb": total_vram,
                        "vram_util_pct": vram_pct,
                        "steps_completed": steps_done,
                        "steps_target": steps_tgt,
                        "duration_sec": duration,
                        "error_message": err_msg,
                    }

                    results_all.append(row_dict)

                    # Append row to CSV and flush immediately
                    with open(output_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([row_dict[h] for h in csv_headers])
                        f.flush()
                        os.fsync(f.fileno())

                    # Console status print
                    if status == "PASS":
                        flag = "[PASS]"
                        details = f"VRAM: {peak_resv}GB / {total_vram}GB ({vram_pct}%) | Alloc: {peak_alloc}GB ({duration}s)"
                    elif status == "OOM":
                        flag = "[OOM] "
                        details = f"Failed at step {steps_done + 1} ({duration}s)"
                    else:
                        flag = "[FAIL]"
                        details = f"{err_msg[:60]} ({duration}s)"

                    print(f"    -> Result: {flag} {details}")

    total_time = round(time.time() - start_all, 1)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"                     GRID SEARCH SUMMARY (Total Time: {total_time}s)")
    print("=" * 90)

    table_headers = ["Task", "Model", "Batch Size", "Hidden Dim", "Status", "Allocated", "Reserved", "VRAM %"]
    table_rows = []
    for r in results_all:
        status_str = r["status"]
        if status_str == "PASS":
            status_str = "PASS" if r["vram_util_pct"] <= 85.0 else "WARN (>85%)"

        table_rows.append([
            r["task"].upper(),
            r["model"],
            str(r["batch_size"]),
            str(r["hidden_size"]),
            status_str,
            f"{r['peak_allocated_gb']:.1f} GB",
            f"{r['peak_reserved_gb']:.1f} / {r['total_vram_gb']:.1f} GB",
            f"{r['vram_util_pct']:.1f}%",
        ])

    col_widths = [len(h) for h in table_headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    sep = "-+-".join(["-" * w for w in col_widths])

    print(fmt.format(*table_headers))
    print(sep)
    for row in table_rows:
        print(fmt.format(*row))

    print("=" * 90)
    print(f"Report saved to CSV: {output_csv.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()

