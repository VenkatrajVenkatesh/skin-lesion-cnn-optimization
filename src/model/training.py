
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any
import os
import json
from evaluation import  evaluate_model_complete

import numpy as np
import time

def calculate_super_aggressive_focal_weights(class_counts_dict):
   
    weights = []
    for class_name, count in class_counts_dict.items():
        if count < 500:  # "too far away"
            weight = 25.0  
        elif count < 2000:  # Small classes
            weight = 10.0  
        elif count < 5000:  # Medium classes
            weight = 5.0   
        else:  # Large classes
            weight = 1.0   
        
        weights.append(weight)
    
    return torch.FloatTensor(weights)

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_composite_score(multiclass_metrics, binary_metrics, class_importance_weights=None):
    
    if class_importance_weights is None:
        class_importance_weights = [0.15, 0.25, 0.25, 0.35]  
    
    per_class_f1 = multiclass_metrics['per_class_f1']
    weighted_f1 = sum(f1 * weight for f1, weight in zip(per_class_f1, class_importance_weights))
    
        # Enhanced composite score calculation
    composite = (
        0.35 * weighted_f1 +                                    
        0.20 * multiclass_metrics['balanced_accuracy'] +       
        0.15 * multiclass_metrics['macro_f1'] +                 
        0.20 * binary_metrics['f1_score'] +                    
        0.10 * min(per_class_f1)                                
    )
    
    return composite

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: torch.device,
    save_path: str = "best_model.pth",
    results_dir: str = "results",
    model_name: str = "model_name_1",
    measure_speed: bool= True,
    speed_measurement_frequency: int =5

    
):
    num_epochs = config['training']['epochs']
    model_name =  config['model']['arch']
    lr = config['training']['lr']
    num_classes = config['model']['num_classes']
    
    # Calculate class counts from training data
    train_df = train_loader.dataset.df
    label_counts = train_df['label'].value_counts()
    
    label_map = config['label_map']
    class_counts = {}
    for label_name, label_id in label_map.items():
        class_counts[label_name] = label_counts.get(label_name, 0)
    
    class_names = list(label_map.keys())
    
    # Calculate Focal Loss weights
    alpha_weights = calculate_super_aggressive_focal_weights(class_counts)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(
        alpha=alpha_weights.to(device),
        gamma=3.0,
        reduction='mean'
    )
    
    print(f"Using Focal Loss with gamma=3.0")
    def calculate_inference_speed(model, dataloader, device, n_batches=10):
        model.eval()
        total_images = 0
        start_time = time.time()
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= n_batches:
                    break
                images = images.to(device)
                _= model(images)
                total_images += images.size(0)
        end_time = time.time()
        speed = total_images / (end_time - start_time)
        return round(speed, 2)
    def compute_val_loss(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                targets = torch.argmax(labels, dim=1)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    model.to(device)
    best_composite_score = 0.0
    all_results = []
    best_model_size_MB = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            targets = torch.argmax(labels, dim=1)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        avg_train_loss = running_loss / len(train_loader)
        measure_speed = measure_speed and (epoch ==0 or (epoch + 1)% speed_measurement_frequency == 0 or epoch == num_epochs -1)

        # Complete Evaluation - Both multiclass and binary
        complete_metrics = evaluate_model_complete(
            model, val_loader, device, 
            epoch=epoch+1, 
            class_names=class_names, 
            verbose=True  
        )
        
        # Extract metrics for compatibility
        multiclass_metrics = complete_metrics['multiclass']
        binary_metrics = complete_metrics['binary']
        
        # Calculate enhanced composite score
        composite_score = calculate_composite_score(multiclass_metrics, binary_metrics)
        val_loss = compute_val_loss( model, val_loader, criterion, device) 
        
        # Store comprehensive results
        result = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "composite_score": composite_score,
            "epoch_time": epoch_time,
            "val_loss": val_loss,
            
            # Multiclass metrics
            "multiclass_accuracy": multiclass_metrics['accuracy'],
            "multiclass_balanced_accuracy": multiclass_metrics['balanced_accuracy'],
            "multiclass_macro_f1": multiclass_metrics['macro_f1'],
            "multiclass_weighted_f1": multiclass_metrics['weighted_f1'],
            "multiclass_macro_precision": multiclass_metrics['macro_precision'],
            "multiclass_macro_recall": multiclass_metrics['macro_recall'],
            "per_class_f1": multiclass_metrics['per_class_f1'],
            "per_class_precision": multiclass_metrics['per_class_precision'],
            "per_class_recall": multiclass_metrics['per_class_recall'],
            "multiclass_confusion_matrix": multiclass_metrics['confusion_matrix'],
            "multiclass_roc_auc": multiclass_metrics['roc_auc_ovr'],
            "per_class_roc_auc": multiclass_metrics['per_class_roc_auc'],
            
            # Binary metrics
            "binary_accuracy": binary_metrics['accuracy'],
            "binary_balanced_accuracy": binary_metrics['balanced_accuracy'],
            "binary_f1": binary_metrics['f1_score'],
            "binary_precision": binary_metrics['precision'],
            "binary_recall": binary_metrics['recall'],
            "binary_specificity": binary_metrics['specificity'],
            "binary_roc_auc": binary_metrics['roc_auc'],
            "binary_confusion_matrix": binary_metrics['confusion_matrix'],

            #Model Data
            "model_name":model_name,
            "model_size_MB": round(best_model_size_MB , 2),
            "inference_speed_img_per_sec": calculate_inference_speed(model,val_loader,device)
        }
        all_results.append(result)

        # Print epoch summary (additional to the detailed evaluation output)
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | val_loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        print(f"Composite Score: {composite_score:.4f}")
        print(f"4-Class Balanced Accuracy: {multiclass_metrics['balanced_accuracy']:.4f}")
        print(f"Binary Accuracy (Useful vs Not-Useful): {binary_metrics['accuracy']:.4f}")
        print(f"Minority Class ({class_names[-1]}) F1: {multiclass_metrics['per_class_f1'][-1]:.4f}")
        

        # Save best model based on composite score
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            torch.save(model.state_dict(), save_path)

            best_model_size_MB = get_model_size(save_path)
            print(f"Best model saved - Composite Score: {best_composite_score:.4f}")

    # Training completion summary
    print(f"\nTRAINING COMPLETE")
    print(f"Best Composite Score: {best_composite_score:.4f}")
    
    # Find and display best epoch results
    best_epoch_result = max(all_results, key=lambda x: x['composite_score'])
    print(f"\nBest Results (Epoch {best_epoch_result['epoch']}):")
    
    print(f"\n4-Class Performance:")
    print(f"  Accuracy: {best_epoch_result['multiclass_accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_epoch_result['multiclass_balanced_accuracy']:.4f}")
    print(f"  Macro F1: {best_epoch_result['multiclass_macro_f1']:.4f}")
    print(f"  Weighted F1: {best_epoch_result['multiclass_weighted_f1']:.4f}")
    print(f"  ROC AUC: {best_epoch_result['multiclass_roc_auc']:.4f}")
    
    print(f"\nBinary Performance (Useful vs Not-Useful):")
    print(f"  Accuracy: {best_epoch_result['binary_accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_epoch_result['binary_balanced_accuracy']:.4f}")
    print(f"  F1 Score: {best_epoch_result['binary_f1']:.4f}")
    print(f"  Precision: {best_epoch_result['binary_precision']:.4f}")
    print(f"  Recall: {best_epoch_result['binary_recall']:.4f}")
    print(f"  Specificity: {best_epoch_result['binary_specificity']:.4f}")
    print(f"  ROC AUC: {best_epoch_result['binary_roc_auc']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for i, class_name in enumerate(class_names):
        if i < len(best_epoch_result['per_class_f1']):
            f1 = best_epoch_result['per_class_f1'][i]
            print(f"  {class_name:15}: {f1:.4f}")

    # Save comprehensive results
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed training log
    with open(os.path.join(results_dir, f"{model_name}_comprehensive_training_log.json"), "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    
    # Save best epoch results separately
    with open(os.path.join(results_dir, f"{model_name}_best_epoch_results.json"), "w") as f:
        json.dump(best_epoch_result, f, indent=4, default=str)
  
    print(f"\nResults saved:")
    print(f"  Comprehensive log: {results_dir}/{model_name}_comprehensive_training_log.json")
    print(f"  Best epoch results: {results_dir}/{model_name}_best_epoch_results.json")
    
    return model