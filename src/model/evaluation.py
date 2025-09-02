import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import time
import numpy as np


class ComprehensiveEvaluator:
   
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None, useful_class_idx: int = 0):
       
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.useful_class_idx = useful_class_idx 
    
    def evaluate_complete(self, model, dataloader, device, epoch: int = 0) -> Dict[str, Any]:

        # Get predictions and targets
        all_preds, all_targets, all_probs = self._get_predictions(model, dataloader, device)
        
        # Compute multiclass metrics
        multiclass_metrics = self._compute_multiclass_metrics(all_targets, all_preds, all_probs)
        
        # Compute binary metrics
        binary_metrics = self._compute_binary_metrics(all_targets, all_preds, all_probs)
        
        # Combine results
        complete_metrics = {
            'epoch': epoch,
            'multiclass': multiclass_metrics,
            'binary': binary_metrics
        }
        
        return complete_metrics
    
    def _get_predictions(self, model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)

#  dequantize if the model returned quantized logits (INT8 path)
                if hasattr(outputs, "is_quantized") and outputs.is_quantized:
                    outputs = outputs.dequantize()

                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                 # Convert one-hot labels to integer labels if necessary
                if labels.dim() > 1:
                    targets = torch.argmax(labels, dim=1)
                else:
                    targets = labels
                
                # Store predictions, targets, and probabilities
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs)
    
    def _compute_multiclass_metrics(self, targets: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
        
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(targets, preds),
            'balanced_accuracy': balanced_accuracy_score(targets, preds),
            
            # Per-class metrics
            'per_class_precision': precision_score(targets, preds, average=None, zero_division=0).tolist(),
            'per_class_recall': recall_score(targets, preds, average=None, zero_division=0).tolist(),
            'per_class_f1': f1_score(targets, preds, average=None, zero_division=0).tolist(),
            
            # Averaged metrics
            'macro_precision': precision_score(targets, preds, average='macro', zero_division=0),
            'macro_recall': recall_score(targets, preds, average='macro', zero_division=0),
            'macro_f1': f1_score(targets, preds, average='macro', zero_division=0),
            
            'weighted_precision': precision_score(targets, preds, average='weighted', zero_division=0),
            'weighted_recall': recall_score(targets, preds, average='weighted', zero_division=0),
            'weighted_f1': f1_score(targets, preds, average='weighted', zero_division=0),
            
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(targets, preds).tolist()
        }
        
        # ROC AUC for multiclass
        try:
            if probs.ndim == 2 and probs.shape[1] == self.num_classes:
                metrics['roc_auc_ovr'] = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
                per_class_auc = []
                for i in range(self.num_classes):
                    try:
                        binary_targets = (targets == i).astype(int)
                        class_probs = probs[:, i]
                        auc = roc_auc_score(binary_targets, class_probs)
                        per_class_auc.append(auc)
                    except ValueError:
                        per_class_auc.append(0.0)
                metrics['per_class_roc_auc'] = per_class_auc
            else:
                # shape mismatch: fall back safely
                metrics['roc_auc_ovr'] = 0.0
                metrics['per_class_roc_auc'] = [0.0] * self.num_classes
        except ValueError:
            metrics['roc_auc_ovr'] = 0.0
            metrics['per_class_roc_auc'] = [0.0] * self.num_classes
        
        # Class support (number of samples per class)
        unique, counts = np.unique(targets, return_counts=True)
        support = np.zeros(self.num_classes)
        for cls, count in zip(unique, counts):
            if cls < self.num_classes:
                support[int(cls)] = count
        metrics['per_class_support'] = support.tolist()
        
        return metrics
    
    def _compute_binary_metrics(self, targets: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
        if probs.ndim != 2 or probs.shape[1] <= self.useful_class_idx:
        
            acc = accuracy_score((targets != self.useful_class_idx).astype(int),
                             (preds   != self.useful_class_idx).astype(int))
            cm = confusion_matrix((targets != self.useful_class_idx).astype(int),
                              (preds   != self.useful_class_idx).astype(int))
    
            if cm.shape != (2, 2):
                full = np.zeros((2,2), dtype=int)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        full[i, j] = cm[i, j]
                cm = full

            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            return {
            'accuracy': acc,
            'balanced_accuracy': balanced_accuracy_score((targets != self.useful_class_idx).astype(int),
                                                         (preds   != self.useful_class_idx).astype(int)),
            'precision': precision_score((targets != self.useful_class_idx).astype(int),
                                         (preds   != self.useful_class_idx).astype(int), zero_division=0),
            'recall': recall_score((targets != self.useful_class_idx).astype(int),
                                   (preds   != self.useful_class_idx).astype(int), zero_division=0),
            'f1_score': f1_score((targets != self.useful_class_idx).astype(int),
                                 (preds   != self.useful_class_idx).astype(int), zero_division=0),
            'confusion_matrix': cm.tolist(),
            'specificity': specificity,
            'true_positives': int(tp), 'true_negatives': int(tn),
            'false_positives': int(fp), 'false_negatives': int(fn),
            'roc_auc': 0.0, 
            'target_distribution': {
                'useful_samples': int(((targets == self.useful_class_idx).sum())),
                'not_useful_samples': int(((targets != self.useful_class_idx).sum()))
            },
            'prediction_distribution': {
                'predicted_useful': int(((preds == self.useful_class_idx).sum())),
                'predicted_not_useful': int(((preds != self.useful_class_idx).sum()))
            }
        }
        
        # Convert to binary
        binary_preds = (preds != self.useful_class_idx).astype(int)
        binary_targets = (targets != self.useful_class_idx).astype(int)
        
        prob_useful = probs[:, self.useful_class_idx]
        prob_not_useful = 1.0 - prob_useful
        # build confusion matrix and force to 2x2 if degenerate
        cm = confusion_matrix(binary_targets, binary_preds)
        if cm.shape != (2, 2):
            full = np.zeros((2,2), dtype=int)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    full[i, j] = cm[i, j]
            cm = full
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics = {
        'accuracy': accuracy_score(binary_targets, binary_preds),
        'balanced_accuracy': balanced_accuracy_score(binary_targets, binary_preds),
        'precision': precision_score(binary_targets, binary_preds, zero_division=0),
        'recall': recall_score(binary_targets, binary_preds, zero_division=0),
        'f1_score': f1_score(binary_targets, binary_preds, zero_division=0),
        'confusion_matrix': cm.tolist(),
        'specificity': specificity,
        'true_positives': int(tp), 'true_negatives': int(tn),
        'false_positives': int(fp), 'false_negatives': int(fn)
    }
        
        
        # ROC AUC for binary
        try:
            metrics['roc_auc'] = roc_auc_score(binary_targets, prob_not_useful)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        
        # Class distribution info
        unique_targets, target_counts = np.unique(binary_targets, return_counts=True)
        unique_preds, pred_counts = np.unique(binary_preds, return_counts=True)
        
        metrics['target_distribution'] = {
            'useful_samples': int(target_counts[0]) if 0 in unique_targets else 0,
            'not_useful_samples': int(target_counts[1]) if 1 in unique_targets else 0
        }
        
        metrics['prediction_distribution'] = {
            'predicted_useful': int(pred_counts[0]) if 0 in unique_preds else 0,
            'predicted_not_useful': int(pred_counts[1]) if 1 in unique_preds else 0
        }
        
        return metrics
    
    def print_complete_results(self, metrics: Dict[str, Any], verbose: bool = True):
        """
        Print both multiclass and binary results
        """
        epoch = metrics.get('epoch', 0)
        multiclass = metrics['multiclass']
        binary = metrics['binary']
        
        print(f"\nEpoch {epoch} Complete Evaluation Results:")
        
        # Multiclass results
        print(f"\n4-Class Results:")
        print(f"  Accuracy: {multiclass['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {multiclass['balanced_accuracy']:.4f}")
        print(f"  Macro F1: {multiclass['macro_f1']:.4f}")
        print(f"  Weighted F1: {multiclass['weighted_f1']:.4f}")
        
        if verbose:
            print(f"  ROC AUC (OvR): {multiclass['roc_auc_ovr']:.4f}")
        
        # Per-class performance
        print(f"\n  Per-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            if i < len(multiclass['per_class_f1']):
                f1 = multiclass['per_class_f1'][i]
                precision = multiclass['per_class_precision'][i]
                recall = multiclass['per_class_recall'][i]
                support = multiclass['per_class_support'][i] if i < len(multiclass['per_class_support']) else 0
                
                print(f"    {class_name:15}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f} (n={support:.0f})")
        
        # Binary results
        print(f"\nBinary Results (Useful vs Not-Useful):")
        print(f"  Accuracy: {binary['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {binary['balanced_accuracy']:.4f}")
        print(f"  Precision: {binary['precision']:.4f}")
        print(f"  Recall: {binary['recall']:.4f}")
        print(f"  Specificity: {binary['specificity']:.4f}")
        print(f"  F1 Score: {binary['f1_score']:.4f}")
        
        if verbose:
            print(f"  ROC AUC: {binary['roc_auc']:.4f}")
            
            
            # Confusion matrix
            cm = binary['confusion_matrix']
            print(f"\n  Binary Confusion Matrix:")
            print(f"                  Predicted")
            print(f"                  Useful  Not-Useful")
            print(f"  Actual Useful     {cm[0][0]:4d}      {cm[0][1]:4d}")
            print(f"      Not-Useful    {cm[1][0]:4d}      {cm[1][1]:4d}")

def evaluate_model_complete(model, dataloader, device, epoch=0, class_names=None, verbose=True):
   
    if class_names is None:
        class_names = ['useful', 'blurry', 'not relevant', 'too far away']
    
    evaluator = ComprehensiveEvaluator(
        num_classes=len(class_names),
        class_names=class_names,
        useful_class_idx=0  
    )
    
    metrics = evaluator.evaluate_complete(model, dataloader, device, epoch)
    
    if verbose:
        evaluator.print_complete_results(metrics, verbose=verbose)
    
    return metrics

def save_evaluation_results(metrics: Dict[str, Any], filepath: str):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4, default=str)
    
    print(f"Evaluation results saved to: {filepath}")
  
def measure_inference_speed(model, dataloader, device_name='cpu', num_batches=10):
   
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    sample_batches = []
    total_images = 0
    for i, (images, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        sample_batches.append(images)
        total_images += images.shape[0]
    if not sample_batches:
        raise ValueError("No batches available for speed measurement")
    # Warmup runs
    with torch.no_grad():
        for images in sample_batches[:min(2, len(sample_batches))]:
            images = images.to(device)
            _ = model(images)
    # Synchronize if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Actual timing
    batch_times = []
    with torch.no_grad():
        start_total = time.time()
        for images in sample_batches:
            images = images.to(device)
            start_batch = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_batch = time.time()
            batch_times.append(end_batch - start_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_total = time.time()
    # Calculate metrics
    total_time = end_total - start_total
    avg_batch_time = np.mean(batch_times)
    avg_per_image_time = total_time / total_images
    fps = total_images / total_time
    return {
        'device': device_name, 'total_time_seconds': total_time, 'avg_batch_time_ms': avg_batch_time * 1000, 'avg_per_image_time_ms': avg_per_image_time * 1000,
        'fps': fps, 'total_images_processed': total_images, 'num_batches_tested': len(batch_times), 'batch_times_ms': [t * 1000 for t in batch_times]
        }