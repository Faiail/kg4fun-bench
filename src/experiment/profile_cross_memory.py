import gc
import json
import sys
import torch
from pathlib import Path
from src.data.utils import BatchKeys
from src.models.utils import ReturnKeys
from src.experiment.utils import ParameterKeys
import src.experiment as exp_pkg


def run_memory_profile(
    parameters: dict,
    cls: str,
    batch_size: int | None = None,
    hidden_size: int | None = None,
    steps: int = 10,
    full_epoch: bool = False,
) -> dict:
    """
    Profiles VRAM usage of a cross-encoding experiment.
    Forces warmup=0 (unfreezing all layers) to test worst-case peak memory.
    Gracefully catches CUDA OutOfMemoryError and reports allocated/reserved VRAM.
    """
    # 1. Apply overrides
    general_params = parameters.get(ParameterKeys.GENERAL, {})
    general_params[ParameterKeys.WARMUP] = 0
    general_params[ParameterKeys.NUM_EPOCHS] = 1
    general_params[ParameterKeys.PBAR] = False
    
    device_str = general_params.get(ParameterKeys.DEVICE, "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        device_str = "cpu"
        general_params[ParameterKeys.DEVICE] = device_str

    if batch_size is not None:
        parameters.setdefault("loader", {})["batch_size"] = batch_size

    if hidden_size is not None:
        model_cfg = parameters.setdefault(ParameterKeys.MODEL, {}).setdefault(ParameterKeys.CFG, {})
        model_cfg["final_hidden_size"] = hidden_size

    effective_bs = parameters.get("loader", {}).get("batch_size", "default")
    effective_hs = parameters.get(ParameterKeys.MODEL, {}).get(ParameterKeys.CFG, {}).get("final_hidden_size", "default")
    config_name = parameters.get("config_name", "unknown")

    print(f"\n==================================================================")
    print(f" PROFILING: {config_name}")
    print(f" Class:      {cls}")
    print(f" Batch Size: {effective_bs}")
    print(f" Hidden Dim: {effective_hs}")
    print(f" Device:     {device_str}")
    print(f" Mode:       Worst-case (all layers unfrozen, no warmup)")
    print(f"==================================================================")

    total_vram_gb = 0.0
    if device_str == "cuda":
        total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        gpu_name = torch.cuda.get_device_name(0)
        print(f" GPU Device: {gpu_name} ({total_vram_gb} GB total VRAM)")

    # 2. Instantiate Run class
    run_cls = getattr(exp_pkg, cls)
    run = run_cls(parameters)

    # Ensure all model parameters (including backbone encoder) are unfrozen
    run.model.train()
    for p in run.model.parameters():
        p.requires_grad = True

    # 3. Reset peak memory tracking
    if device_str == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_loader = run.train_loader
    total_batches = len(train_loader)
    steps_to_run = total_batches if full_epoch or steps <= 0 else min(steps, total_batches)
    print(f" Running {steps_to_run} training batch(es)...")

    is_complete_run = (cls == "SharedCrossTextCompleteRun")
    completed_steps = 0
    oom_occurred = False
    error_message = ""

    try:
        for step, batch in enumerate(train_loader):
            if step >= steps_to_run:
                break

            run.optimizer.zero_grad()

            if is_complete_run:
                # 6 backbone encodings + 3 cross units
                input_head_tok = run.tokenize(batch[BatchKeys.INPUT_HEAD])
                input_tail_tok = run.tokenize(batch[BatchKeys.INPUT_TAIL])
                input_rel_tok = run.tokenize(batch[BatchKeys.INPUT_REL])

                target_head_tok = run.tokenize(batch[BatchKeys.TARGET_HEAD])
                target_tail_tok = run.tokenize(batch[BatchKeys.TARGET_TAIL])
                target_rel_tok = run.tokenize(batch[BatchKeys.TARGET_REL])

                scores = run.model(
                    input_head=input_head_tok,
                    input_tail=input_tail_tok,
                    input_rel=input_rel_tok,
                    target_head=target_head_tok,
                    target_tail=target_tail_tok,
                    target_rel=target_rel_tok,
                )
                labels = batch.get(
                    BatchKeys.GT,
                    {k: torch.arange(v.size(0)).long() for k, v in scores.items()},
                )
                labels = {k: v.to(run.device) for k, v in labels.items()}
                loss = run.criterion(scores, labels)
            else:
                # Node / Edge: 2 backbone encodings + 1 cross unit
                input_tok = run.tokenize(batch[BatchKeys.INPUT])
                schema_tok = run.tokenize(batch[BatchKeys.SCHEMA])
                scores = run.model(input_tok, schema_tok)
                labels = batch.get(BatchKeys.GT, torch.arange(scores.size(0)).long()).to(run.device)
                loss = run.criterion(scores, labels)

            loss.backward()
            run.optimizer.step()
            if run.scheduler is not None:
                run.schedule(phase=ParameterKeys.TRAIN)

            completed_steps += 1

            if device_str == "cuda":
                alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"   Batch {step + 1}/{steps_to_run} - loss: {loss.item():.4f} | alloc: {alloc_gb:.2f}GB | reserved: {res_gb:.2f}GB")
            else:
                print(f"   Batch {step + 1}/{steps_to_run} - loss: {loss.item():.4f}")

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_str = str(e)
        if "out of memory" in err_str.lower() or "cuda" in err_str.lower() and "memory" in err_str.lower():
            oom_occurred = True
            error_message = err_str.split("\n")[0]
            print(f"\n [!!!] CUDA OUT OF MEMORY triggered at batch {completed_steps + 1}!")
            print(f"       Details: {error_message}")
        else:
            raise e
    finally:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0
        pct_used = 0.0

        if device_str == "cuda":
            peak_allocated_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
            peak_reserved_gb = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
            if total_vram_gb > 0:
                pct_used = round((peak_reserved_gb / total_vram_gb) * 100, 1)

            # Cleanup VRAM for caller
            torch.cuda.empty_cache()
            gc.collect()

    status = "OOM" if oom_occurred else "PASS"
    result = {
        "config_name": config_name,
        "class": cls,
        "batch_size": effective_bs,
        "hidden_size": effective_hs,
        "steps_completed": completed_steps,
        "steps_target": steps_to_run,
        "status": status,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "total_vram_gb": total_vram_gb,
        "pct_used": pct_used,
        "error_message": error_message,
    }

    print("\n------------------------------------------------------------------")
    print(f" STATUS:          {status}")
    print(f" Peak Allocated:  {peak_allocated_gb} GB")
    print(f" Peak Reserved:   {peak_reserved_gb} GB / {total_vram_gb} GB ({pct_used}%)")
    print(f" Batches Run:     {completed_steps}/{steps_to_run}")
    print("------------------------------------------------------------------")
    print(f"PROFILE_RESULT: {json.dumps(result)}")

    return result

