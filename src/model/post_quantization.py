import os
import time
import json
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,balanced_accuracy_score,precision_score, recall_score
from utils import load_config, load_and_process_csv
from models import create_model
from dataloaders import get_dataloaders
config = load_config()
model_name = config['model']['arch']

def get_model_size(filepath):
    """Calculate model file size in MB"""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def export_torchscript(model, path, img_size=224, dtype=torch.float32, device="cpu"):
    model = model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device)
    example = torch.randn(1, 3, img_size, img_size, device=device)

    # Only cast to fp16 on CUDA; CPU half often unsupported for ops
    if dtype == torch.float16:
        if device == "cuda":
            model = model.half()
            example = example.half()
        else:
            dtype = torch.float32  # fallback to fp32

    with torch.inference_mode():
        try:
            scripted = torch.jit.trace(model, example)
        except Exception:
            scripted = torch.jit.script(model)

    scripted.save(path)
    size_mb = get_model_size(path)
    return scripted, size_mb, path


def save_state_dict(model, path):
    model = model.eval().cpu()
    torch.save(model.state_dict(), path)
    size_mb = get_model_size(path)
    return model, size_mb, path

# Quantization variants

def save_fp32_torchscript(model, save_prefix="model_fp32", img_size=224, device="cpu"):
    path = f"{model_name}_{save_prefix}.pt"
    art, size, _ = export_torchscript(model, path, img_size=img_size, dtype=torch.float32, device=device)
    print(f"FP32 TorchScript: {path} ({size:.1f} MB)")
    return art, size


def save_fp16_torchscript(model, save_prefix="model_fp16", img_size=224, device="cuda"):
    # fp16 TorchScript only meaningful on CUDA; otherwise falls back to fp32 silently
    path = f"{model_name}_{save_prefix}.pt"
    art, size, _ = export_torchscript(model, path, img_size=img_size, dtype=torch.float16, device=device)
    tag = "FP16 TorchScript (CUDA)" if (device == "cuda" and torch.cuda.is_available()) else "FP16 requested -> saved as FP32 on CPU"
    print(f"{tag}: {path} ({size:.1f} MB)")
    return art, size


def post_quant_int8_dynamic_torchscript(model, save_prefix="model_int8_dynamic", img_size=224, device="cpu"):
    model = model.eval().cpu()
    # Quantize only Linear layers dynamically (as designed)
    try:
        qdyn = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except AttributeError:
        qdyn = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    # Save as TorchScript so packed int8 weights are reflected on disk size
    path = f"{model_name}_{save_prefix}.pt"
    art, size, _ = export_torchscript(qdyn, path, img_size=img_size, dtype=torch.float32, device=device)
    print(f"INT8 Dynamic TorchScript: {path} ({size:.1f} MB)")
    return art, size, qdyn



def print_quantized_linear_layers(model_before, model_after):
    print("Quantized Linear layers:")
    count = 0
    for (n1, m1), (n2, m2) in zip(model_before.named_modules(), model_after.named_modules()):
        if isinstance(m1, nn.Linear) and ("quantized" in m2.__class__.__module__ or "quantized" in m2.__class__.__name__.lower()):
            print(f"  {n2}: {m2.__class__.__name__}")
            count += 1
    if count == 0:
        print("  None")


def _safe_param_dtype(model, default=torch.float32):
    try:
        return next(model.parameters()).dtype
    except Exception:
        return default


def test_inference_time(model, img_size=224, device="cpu", runs=100):
    try:
        model_dtype = _safe_param_dtype(model, default=torch.float32)

        if device == "cpu" and model_dtype == torch.float16 and hasattr(model, "float"):
            model = model.float()

        model = model.eval()
        x = torch.randn(1, 3, img_size, img_size)

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            model = model.to("cuda")
            x = x.to("cuda")
            if _safe_param_dtype(model) == torch.float16:
                x = x.half()

        with torch.inference_mode():
            for _ in range(10):
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with torch.inference_mode():
            for _ in range(runs):
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        avg_ms = (time.time() - start) / runs * 1000.0
        fps = 1000.0 / avg_ms
        return avg_ms, fps
    except Exception as e:
        print(f"Speed test failed on {device}: {e}")
        return None, None


def evaluate_model(model, val_loader, device, model_name):
    if val_loader is None:
        return None

    try:
        model_dtype = _safe_param_dtype(model, default=torch.float32)
        if model_dtype == torch.float16 and device.type == "cpu":
            model = model.float()

        model = model.to(device) if hasattr(model, "to") else model
        model.eval()

        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                if _safe_param_dtype(model) == torch.float16:
                    images = images.half()
                else:
                    images = images.float()

                outputs = model(images)
                if outputs.dtype == torch.float16:
                    outputs = outputs.float()

                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                if labels.dim() > 1:
                    targets = torch.argmax(labels, dim=1)
                else:
                    targets = labels

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        binary_preds = (all_preds != 0).astype(int)
        binary_targets = (all_targets != 0).astype(int)
        f1 = f1_score(binary_targets, binary_preds, zero_division=0)

        metrics = {"multiclass": {"accuracy": acc},
                     
                     "balanced_accuracy": balanced_acc,
                     "precision": precision,
                     "recall": recall,
                     "binary": {"f1_score": f1}}
        print(f"{model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, Balanced Acc: {balanced_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return metrics
    except Exception as e:
        print(f"Evaluation failed for {model_name}: {e}")
        return None



def create_model_copy(original_model, config):
    new_model = create_model(config)
    new_model.load_state_dict(copy.deepcopy(original_model.state_dict()))
    new_model.eval()
    return new_model


def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPerformance Metrics:")
    print(f"{'Model':<12} {'Accuracy':<10} {'Binary F1':<10} {'Size (MB)':<12} {'CPU (ms)':<10}")
    print("-" * 60)

    for key in ["fp32", "fp16", "int8_dynamic"]:
        if key not in results or results[key] is None:
            continue
        data = results[key]
        metrics = data.get("metrics")
        acc = metrics["multiclass"]["accuracy"] if metrics else "N/A"
        f1 = metrics["binary"]["f1_score"] if metrics else "N/A"
        size = data.get("size_mb")
        cpu_time = data.get("cpu_time_ms")

        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        size_str = f"{size:.1f}" if isinstance(size, (int, float)) else str(size)
        cpu_str = f"{cpu_time:.1f}" if isinstance(cpu_time, (int, float)) else str(cpu_time)

        print(f"{key.upper():<12} {acc_str:<10} {f1_str:<10} {size_str:<12} {cpu_str:<10}")

    print(f"\nOptimization Gains vs FP32:")
    fp32 = results.get("fp32", {})
    fp32_acc = fp32.get("metrics", {}).get("multiclass", {}).get("accuracy", None)
    fp32_time = fp32.get("cpu_time_ms", None)
    fp32_size = fp32.get("size_mb", None)

    for key in ["fp16", "int8_dynamic"]:
        data = results.get(key, None)
        if not data:
            continue
        print(f"{key.upper()}: Size reduction {data.get('size_reduction', 'N/A')}")
        if fp32_acc and data.get("metrics"):
            acc_drop = ((fp32_acc - data["metrics"]["multiclass"]["accuracy"]) / fp32_acc) * 100
            print(f"           Accuracy drop: {acc_drop:.2f}%")
        if fp32_time and data.get("cpu_time_ms"):
            speedup = fp32_time / data["cpu_time_ms"]
            print(f"           CPU speedup: {speedup:.1f}x")





def comprehensive_quantization_benchmark(config, weights_path=None):
    print("QUANTIZATION BENCHMARK")
    print("=" * 50)

    img_size = int(config.get("img_size", 224))
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

    # Load base model
    base_model = create_model(config)
    if weights_path is None:
        weights_path = config.get("weights_path", None)
    if weights_path is None or not os.path.exists(weights_path):
        print(f"Model weights not found: {weights_path}")
        return None

    base_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    base_model.eval()

    
    try:
        df = load_and_process_csv(config["csv_path"])
        _, val_loader = get_dataloaders(df, config)
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Could not load validation data: {e}")
        val_loader = None

    results = {
        "architecture": config["model"]["arch"],
        "validation_samples": len(val_loader.dataset) if val_loader else 0,
    }

    # 1) FP32
    print("\n1. FP32 Baseline")
    fp32_art, fp32_size = save_fp32_torchscript(create_model_copy(base_model, config), "model_fp32", img_size, device="cpu")
    fp32_cpu_time, fp32_cpu_fps = test_inference_time(create_model_copy(base_model, config), img_size=img_size, device="cpu")
    if device_gpu:
        fp32_gpu_time, fp32_gpu_fps = test_inference_time(create_model_copy(base_model, config), img_size=img_size, device="cuda")
    else:
        fp32_gpu_time, fp32_gpu_fps = None, None
    fp32_metrics = evaluate_model(create_model_copy(base_model, config), val_loader, device_gpu or device_cpu, "FP32") if val_loader else None

    results["fp32"] = {
        "size_mb": fp32_size,
        "cpu_time_ms": fp32_cpu_time,
        "cpu_fps": fp32_cpu_fps,
        "gpu_time_ms": fp32_gpu_time,
        "gpu_fps": fp32_gpu_fps,
        "metrics": fp32_metrics,
    }

    # 2) FP16 TorchScript (CUDA if available; CPU fallback = fp32)
    print("\n2. FP16")
    fp16_art, fp16_size = save_fp16_torchscript(create_model_copy(base_model, config), "model_fp16", img_size, device="cuda" if device_gpu else "cpu")
    fp16_cpu_time, fp16_cpu_fps = test_inference_time(create_model_copy(base_model, config).half(), img_size=img_size, device="cpu")
    if device_gpu:
        fp16_gpu_time, fp16_gpu_fps = test_inference_time(create_model_copy(base_model, config).half(), img_size=img_size, device="cuda")
    else:
        fp16_gpu_time, fp16_gpu_fps = None, None
    fp16_metrics = evaluate_model(create_model_copy(base_model, config).half(), val_loader, device_gpu or device_cpu, "FP16") if val_loader else None

    results["fp16"] = {
        "size_mb": fp16_size,
        "cpu_time_ms": fp16_cpu_time,
        "cpu_fps": fp16_cpu_fps,
        "gpu_time_ms": fp16_gpu_time,
        "gpu_fps": fp16_gpu_fps,
        "metrics": fp16_metrics,
        "size_reduction": f"{((fp32_size - fp16_size) / fp32_size * 100):.1f}%" if isinstance(fp32_size, (int, float)) else "N/A",
    }

    # 3) INT8 Dynamic (TorchScript)
    print("\n3. INT8 Dynamic")
    int8_art, int8_size, int8_qdyn = post_quant_int8_dynamic_torchscript(create_model_copy(base_model, config), "model_int8_dynamic", img_size, device="cpu")
    print_quantized_linear_layers(create_model_copy(base_model, config), int8_qdyn)
    int8_cpu_time, int8_cpu_fps = test_inference_time(int8_qdyn, img_size=img_size, device="cpu")
    int8_metrics = evaluate_model(int8_qdyn, val_loader, device_cpu, "INT8-Dynamic") if val_loader else None

    results["int8_dynamic"] = {
        "size_mb": int8_size,
        "cpu_time_ms": int8_cpu_time,
        "cpu_fps": int8_cpu_fps,
        "metrics": int8_metrics,
        "size_reduction": f"{((fp32_size - int8_size) / fp32_size * 100):.1f}%" if isinstance(fp32_size, (int, float)) else "N/A",
    }

    print_summary(results)

    results_dir = config['save_post_quant_result_path']
    model_name = config['model']['arch']
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, model_name+"_quantization_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {filepath}")
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional; uses utils.load_config default if not given)")
    ap.add_argument("--weights", type=str, default='filepath', help="Path to trained FP32 weights (.pth)")
    return ap.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        cfg = load_config() if args.config is None else load_config(args.config)
        print(f"Config loaded for {cfg['model']['arch']}")
        results = comprehensive_quantization_benchmark(cfg, weights_path=args.weights or cfg.get("weights_path"))
        if results:
            print("\nQuantization  completed successfully.")
        else:
            print("\nQuantization  failed.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
