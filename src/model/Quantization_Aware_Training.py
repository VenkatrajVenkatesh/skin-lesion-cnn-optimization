import os
import time
import json
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import statistics


from torch.ao.quantization import (
    prepare_qat,
    convert,
    QConfig,
)
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,       
    PerChannelMinMaxObserver           
)
from torch.ao.quantization import QuantStub, DeQuantStub

from utils import load_config, load_and_process_csv
from dataloaders import get_dataloaders
from models import create_model
from training import FocalLoss, calculate_super_aggressive_focal_weights, calculate_composite_score
from evaluation import evaluate_model_complete


# Ensure FBGEMM backend (x86)
torch.backends.quantized.engine = "fbgemm"

def _set_qconfig_none_recursive(m: nn.Module):
    m.qconfig = None
    for c in m.children():
        _set_qconfig_none_recursive(c)

def make_fbgemm_qat_qconfig() -> QConfig:

    return QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
        ),
    )
import math
import statistics



class QATImageClassifier(nn.Module):
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
       
        for attr in ("config", "arch", "num_classes", "dropout", "pretrained", "feature_dim"):
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

        
        self.backbone = base_model.backbone
        _set_qconfig_none_recursive(self.backbone)   
        self.classifier = base_model.classifier

        self.backbone.qconfig = None  
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

       
        self.qconfig = make_fbgemm_qat_qconfig()

    def forward(self, x):
        
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        qfeats =self.quant(feats)
        logits = self.classifier(qfeats)
        return self.dequant(logits)


def prepare_qat_model(fp32_model: nn.Module) -> nn.Module:
    print("Preparing model for QAT (FBGEMM, per_channel_symmetric weights)...")
    model = QATImageClassifier(fp32_model.cpu()).train()
    model = prepare_qat(model)
    print("Prepared for QAT.")
    return model


def convert_qat_to_int8(qat_model: nn.Module, save_path: str) -> nn.Module:
    print("Converting to INT8...")
    torch.backends.quantized.engine = 'fbgemm'
    model_1 = qat_model.eval()
    int8_model_1 = convert(model_1, inplace=False)
    sd_path = save_path.replace(".pth", "_state_dict.pth")

    

    sd = int8_model_1.state_dict()
    torch.save(sd, sd_path)
    size_mb = os.path.getsize(sd_path) / (1024 * 1024)
    print(f"Saved INT8 state_dict: {save_path} ({size_mb:.2f} MB)")
    
    qat_model = qat_model.cpu().eval()
    int8_model = convert(qat_model, inplace=False)
    torch.save(int8_model, save_path)  # save full module (simpler than state_dict)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved INT8 model: {save_path} ({size_mb:.2f} MB)")
    return int8_model



def train_qat_model(
    qat_model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: torch.device,
    save_path: str = "best_qat_model.pth",
    results_dir: str = "qat_results"
) -> nn.Module:
    
    os.makedirs(results_dir, exist_ok=True)

    qat_epochs = int(config.get("qat_epochs", 5))
    base_lr = float(config["training"]["lr"])
    qat_lr = float(config.get("qat_lr", base_lr * 0.1))

    print("=" * 60)
    print("QUANTIZATION-AWARE TRAINING")
    print(f"QAT epochs:        {qat_epochs}")
    print(f"QAT learning rate: {qat_lr}")
    print("=" * 60)

   
    train_df = train_loader.dataset.df
    label_counts = train_df["label"].value_counts()
    label_map = config["label_map"]
    class_names = list(label_map.keys())
    class_counts = {name: label_counts.get(name, 0) for name in class_names}

    alpha_weights = calculate_super_aggressive_focal_weights(class_counts)

    optimizer = optim.Adam(qat_model.parameters(), lr=qat_lr)
    criterion = FocalLoss(alpha=alpha_weights.to(device), gamma=3.0, reduction="mean")

    qat_model.to(device).train()

    best_composite_score = float("-inf")
    all_results = []

    for epoch in range(qat_epochs):
        start = time.time()
        qat_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            targets = torch.argmax(labels, dim=1) 

            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start
        avg_train_loss = running_loss / max(1, len(train_loader))

        
        qat_model.eval()
        metrics = evaluate_model_complete(
            qat_model, val_loader, device,
            epoch=epoch + 1, class_names=class_names, verbose=False
        )
        mc = metrics["multiclass"]
        bn = metrics["binary"]
        composite = calculate_composite_score(mc, bn)

        all_results.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "composite_score": composite,
            "epoch_time": epoch_time,
            "training_type": "qat",
            "multiclass_accuracy": mc["accuracy"],
            "multiclass_balanced_accuracy": mc["balanced_accuracy"],
            "multiclass_macro_f1": mc["macro_f1"],
            "binary_accuracy": bn["accuracy"],
            "binary_f1": bn["f1_score"],
            "per_class_f1": mc["per_class_f1"],
        })

        print(f"\nEpoch {epoch + 1}/{qat_epochs} | "
              f"train_loss={avg_train_loss:.4f} | time={epoch_time:.1f}s | "
              f"Composite={composite:.4f} | 4-class BalAcc={mc['balanced_accuracy']:.4f} | BinF1={bn['f1_score']:.4f}")

        if composite > best_composite_score:
            best_composite_score = composite
            torch.save(qat_model.state_dict(), save_path)
            print(f"Saved best QAT model (Composite={best_composite_score:.4f})")
    model_name = config['model']['arch']
    
    
    with open(os.path.join(results_dir, model_name+"_qat_training_log.json"), "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    best_result = max(all_results, key=lambda r: r["composite_score"])
    with open(os.path.join(results_dir, model_name+"_qat_best_results.json"), "w") as f:
        json.dump(best_result, f, indent=4, default=str)

    print("\nQAT training complete.")
    print(f"Best composite score: {best_composite_score:.4f}")
    return qat_model


def _normalize_device(dev):
    if isinstance(dev, torch.device):
        return dev.type  # 'cpu' or 'cuda'
    if isinstance(dev, str):
        return dev.lower()
    return "cpu"

def _is_quantized_model(m: torch.nn.Module) -> bool:
    return any(("quantized" in mod.__class__.__module__) or
               mod.__class__.__name__.startswith("Quantized")
               for mod in m.modules())

def _safe_param_dtype(m: torch.nn.Module, default=torch.float32):
    try:
        return next(m.parameters()).dtype
    except StopIteration:
        return default

def test_inference_time(model, img_size=224, device="cpu", runs=100, warmup=10):
   
  
    try:
        device_str = _normalize_device(device)

        if _is_quantized_model(model) and device_str != "cpu":
            raise RuntimeError("Quantized INT8 models must run on CPU in PyTorch eager mode.")

        model_dtype = _safe_param_dtype(model, default=torch.float32)
        model = model.eval()

        # Create input
        x = torch.randn(1, 3, img_size, img_size)

        if device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            model = model.to("cuda")                         
            x = x.to("cuda")                                 
            if model_dtype == torch.float16:
                x = x.half()
        else:
            model = model.to("cpu")                          
            if hasattr(model, "float"):
                model = model.float()
            x = x.to("cpu").float()                          

        # Warm-up
        with torch.inference_mode():
            if device_str == "cuda":
                for _ in range(max(1, warmup)):
                    _ = model(x)
                torch.cuda.synchronize()
            else:
                for _ in range(max(1, warmup)):
                    _ = model(x)

        # Measure
        if device_str == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            with torch.inference_mode():
                for _ in range(runs):
                    _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(runs):
                    _ = model(x)
            total_ms = (time.perf_counter() - t0) * 1000.0

        avg_ms = total_ms / runs
        fps = 1000.0 / avg_ms if avg_ms > 0 else None
        return round(avg_ms, 3), None if fps is None else round(fps, 1)

    except Exception as e:
        print(f"Speed test failed on {device}: {e}")
        return None, None
    


def evaluate_all(
    original_model: nn.Module,
    qat_model: nn.Module,
    quantized_model: nn.Module,
    val_loader,
    device: torch.device,
    class_names: list,
    speed_warmup: int = 5,
    speed_measure: int = 100,
) -> Dict[str, Any]:
   
   
    print("EVALUATING MODELS\n")
    

    results = {}

    print("Evaluating original FP32...")
    original_model = original_model.to(device).eval()
    results["original_fp32"] = evaluate_model_complete(
        original_model, val_loader, device, epoch="ORIGINAL", class_names=class_names, verbose=True
    )
    
    dev_for_fp32 = "cuda" if torch.cuda.is_available() else "cpu"
    fp32_gpu_ms,fp32_gpu_fps = test_inference_time(original_model, 224, device=dev_for_fp32)
    fp32_cpu_ms,fp32_cpu_fps = test_inference_time(original_model, 224, device="cpu")
    if fp32_gpu_ms is None or fp32_cpu_ms is None:
        print("Original FP32 speed test failed (see message above).")
    results["original_fp32"]["speed_cpu"] = {"device": "cpu", "avg_ms": fp32_cpu_ms, "fps": fp32_cpu_fps}
    results["original_fp32"]["speed_gpu"] = {"device": dev_for_fp32, "avg_ms": fp32_gpu_ms, "fps": fp32_gpu_fps}
    print(f"Original FP32 speed on {dev_for_fp32}: {fp32_gpu_fps:.1f} img/s (avg latency {fp32_gpu_ms:.1f} ms)")


    print("\nEvaluating QAT FP32...")
    qat_model = qat_model.to(device).eval()
    results["qat_fp32"] = evaluate_model_complete(
        qat_model, val_loader, device, epoch="QAT", class_names=class_names, verbose=True
    )

    cpu_ms, cpu_fps = test_inference_time(qat_model, 224, device="cpu")
    results["qat_fp32"]["speed_cpu"] = {"device": "cpu", "avg_ms": cpu_ms, "fps": cpu_fps}
    if cpu_ms is not None:
        print(f"QAT FP32 speed on CPU: {cpu_fps:.1f} img/s (avg latency {cpu_ms:.1f} ms)")
    else:
        print("QAT FP32 speed on CPU: timing failed (see message above).")
    gpu_ms, gpu_fps = test_inference_time(qat_model, 224, device="cuda")
    results["qat_fp32"]["speed_gpu"] = {"device": "cuda", "avg_ms": gpu_ms, "fps": gpu_fps}
    if gpu_ms is not None:
        print(f"QAT FP32 speed on CUDA: {gpu_fps:.1f} img/s (avg latency {gpu_ms:.1f} ms)")
    else:
        print("QAT FP32 speed on CUDA: timing failed (see message above).")    
    
    
    print("\nEvaluating INT8 quantized (CPU)...")
    cpu = torch.device("cpu")
    quantized_model = quantized_model.to(cpu).eval()
    cpu_val_loader = torch.utils.data.DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=0
    )
    results["quantized_int8"] = evaluate_model_complete(
        quantized_model, cpu_val_loader, cpu, epoch="QUANTIZED", class_names=class_names, verbose=True
    )
    int8_cpu_ms, int8_cpu_fps = test_inference_time(quantized_model, 224, device="cpu")
    results["quantized_int8"]["speed_CPU"] = {"device": cpu, "avg_ms": int8_cpu_ms, "fps": int8_cpu_fps}
    if int8_cpu_ms is not None:
        print(f"INT8 quantized speed on CPU: {int8_cpu_fps:.1f} img/s (avg latency {int8_cpu_ms:.1f} ms)")
    else:
        print("INT8 quantized speed on CPU: timing failed (see message above).")
    

    print("\nSummary (acc / macroF1 / binF1):")
    for key in ["original_fp32", "qat_fp32", "quantized_int8"]:
        mc = results[key]["multiclass"]
        bn = results[key]["binary"]
        print(f"{key:15s} acc={mc['accuracy']:.4f}  macroF1={mc['macro_f1']:.4f}  binF1={bn['f1_score']:.4f}")

    print("\n=== Single-image latency/FPS ===")
    def _show_speed(block, cpu_key="speed_cpu", gpu_key="speed_gpu"):
        cpu = results[block].get(cpu_key)
        gpu = results[block].get(gpu_key)
        if cpu:
            print(f"{block:15s} CPU  : {cpu['avg_ms']} ms  | {cpu['fps']} FPS")
        if gpu:
            print(f"{'':15s} GPU  : {gpu['avg_ms']} ms  | {gpu['fps']} FPS")

    _show_speed("original_fp32")
    _show_speed("qat_fp32")
    _show_speed("quantized_int8", cpu_key="speed_CPU", gpu_key=None)

    return results




def run_complete_qat_pipeline(
    pretrained_model_path: str,
    config_path: Optional[str] = None,
    output_dir: str = "qat_pipeline_results"
) -> Dict[str, Any]:
    
   
    print("QAT PIPELINE\n")
    

    config = load_config(config_path) if config_path else load_config()
    model_name = config["model"]["arch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Data
    df = load_and_process_csv(config["csv_path"])
    train_loader, val_loader = get_dataloaders(df, config)
    class_names = list(config["label_map"].keys())

    # Original FP32 model
    print(f"Loading FP32 model weights from: {pretrained_model_path}")
    original_model = create_model(config)
    original_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))

    #  Prepare QAT model (from original FP32)
    qat_model = prepare_qat_model(original_model)
    model_name= config['model']['arch']
    # Train QAT
    qat_model_path = os.path.join(output_dir, model_name+"_best_qat_model.pth")
    trained_qat_model = train_qat_model(
        qat_model=qat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_path=qat_model_path,
        results_dir=output_dir
    )

    # Fresh wrapper + quick calibration + convert to INT8
   
    fresh = prepare_qat_model(original_model)

    # Load the trained QAT weights (ignore observer/fake-quant buffer mismatches)
    missing, unexpected = fresh.load_state_dict(trained_qat_model.state_dict(), strict=False)
    print("Loaded QAT weights into fresh wrapper | missing:", len(missing), "| unexpected:", len(unexpected))

   
    #  CPU (FBGEMM is a CPU backend)
    fresh.to("cpu").train()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            _ = fresh(imgs)  
            if i >= 200:     
                break

    # Convert to INT8
    quantized_model_path = os.path.join(output_dir, model_name+"_final_quantized_int8_model.pth")
    final_quantized_model = convert_qat_to_int8(fresh,quantized_model_path)

    #  Evaluate all variants
    eval_results = evaluate_all(
        original_model=original_model,
        qat_model=trained_qat_model,
        quantized_model=final_quantized_model,
        val_loader=val_loader,
        device=device,
        class_names=class_names,
        speed_warmup=5,
        speed_measure=100,
    )
    model_name= config['model']['arch']
    # Save evaluation and summary
    with open(os.path.join(output_dir, model_name+"_qat_evaluation_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4, default=str)

    summary = {
        "original_model_path": pretrained_model_path,
        "qat_model_path": qat_model_path,
        "quantized_model_path": quantized_model_path,
        "output_directory": output_dir
    }
    with open(os.path.join(output_dir, model_name+"_qat_pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=str)

    print("QAT pipeline complete.")
    print(f"INT8 model: {quantized_model_path}")
    return summary




if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qat_pipeline.py <pretrained_model_path> [config_path] [output_dir]")
        sys.exit(1)

    pretrained_model_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "qat_pipeline_results"

    if not os.path.exists(pretrained_model_path):
        print(f"Pretrained model not found: {pretrained_model_path}")
        sys.exit(1)

    run_complete_qat_pipeline(pretrained_model_path, config_path, output_dir)

