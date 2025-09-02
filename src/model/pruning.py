import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from utils import load_config, load_and_process_csv
from dataloaders import get_dataloaders
from models import create_model
from evaluation import evaluate_model_complete
from shrink_head import shrink_classifier_head


config =load_config()

def get_model_size(filepath):
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0
    return sparsity, zero_params, total_params

def apply_structured_pruning(model, pruning_ratio=0.3, num_classes=4):
    print(f"Applying structured pruning with ratio: {pruning_ratio}")
    pruned_modules = []

    # detect arch
    arch = getattr(model, "arch", "")

    for name, module in model.named_modules():
            # Skip classifier layers
        if name.startswith("classifier"):
            continue

        if isinstance(module, nn.Conv2d):
            # OK for all arches
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
            pruned_modules.append((module, 'weight'))
            print(f"  Pruned {name}: {tuple(module.weight.shape)}")

        elif isinstance(module, nn.Linear):
            # 1) Never prune the first linear layer
            if "mobilenetv3" in arch:
                continue
            # 2) Never prune the final classification layer
            out_features = module.weight.shape[0]
            if out_features <= num_classes:
                continue
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
            pruned_modules.append((module, 'weight'))
            print(f"  Pruned {name}: {tuple(module.weight.shape)}")

    return pruned_modules



def apply_global_magnitude_pruning(model, pruning_ratio=0.4):
    print(f"Applying global magnitude pruning with ratio: {pruning_ratio}")
    
   
    modules_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))
    
    # Apply global pruning
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio
    )
    
    print(f"  Applied global pruning to {len(modules_to_prune)} modules")
    
    return modules_to_prune

def fine_tune_pruned_model(model, train_loader, val_loader, config, device, epochs=10):
    print(f"Fine-tuning pruned model for {epochs} epochs...")
    
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr']) 
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(epochs):
        print(f"\nFine-tuning Epoch {epoch+1}/{epochs}")
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Convert one-hot to class indices
            if labels.dim() > 1:
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        train_accuracy = correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"  Training - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, Time: {epoch_time:.1f}s")
        
        # Quick validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if labels.dim() > 1:
                    targets = torch.argmax(labels, dim=1)
                else:
                    targets = labels
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = val_correct / val_total
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'epoch_time': epoch_time
        })
        
        model.train()
    
    print(f"Fine-tuning completed. Best accuracy: {best_accuracy:.4f}")
    
    return model, training_history

def make_pruning_permanent(model):

    print("Making pruning permanent...")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
                print(f"  Made permanent: {name} weight")
            except:
                pass
            
            try:
                prune.remove(module, 'bias')
                print(f"  Made permanent: {name} bias")
            except:
                pass
    
    return model

def evaluate_pruned_model(model, val_loader, device):
    print("\nEvaluating pruned model...")
    try:
        metrics = evaluate_model_complete(
            model, val_loader, device,
            epoch="PRUNED_MODEL",
            class_names=['useful', 'blurry', 'not relevant', 'too far away'],
            verbose=False
        )
        
        print(f"Pruned Model - Accuracy: {metrics['multiclass']['accuracy']:.4f}, "
              f"Balanced Acc: {metrics['multiclass']['balanced_accuracy']:.4f}, "
              f"Binary F1: {metrics['binary']['f1_score']:.4f}")
        
        return metrics
    except Exception as e:
        print(f"Pruned model evaluation failed: {e}")
        # DEBUG: print actual logits shape to confirm the issue
        try:
            model.eval()
            images, _ = next(iter(val_loader))
            images = images.to(device)
            with torch.no_grad():
                outs = model(images)
            print(f"[DEBUG] First-batch logits shape: {tuple(outs.shape)}")
        except Exception as ee:
            print(f"[DEBUG] Could not fetch logits shape: {ee}")
        return None


def comprehensive_pruning_pipeline(trained_model_path, config, 
                                 pruning_method='global', 
                                 pruning_ratio=0.4, 
                                 fine_tune_epochs=3):
   
    
    print("COMPREHENSIVE MODEL PRUNING")
    print(f"Model Path: {trained_model_path}")
    print(f"Pruning Method: {pruning_method}")
    print(f"Pruning Ratio: {pruning_ratio}")
    print(f"Fine-tune Epochs: {fine_tune_epochs}")
    
    if not os.path.exists(trained_model_path):
        print(f"[ERROR] Model weights not found at {trained_model_path}")
        return None
    
    # Extract architecture
    model_filename = os.path.basename(trained_model_path)
    if 'mobilenetv3_large' in model_filename:
        architecture_name = 'mobilenetv3_large'
    elif 'mobilenetv3_small' in model_filename:
        architecture_name = 'mobilenetv3_small'
    elif 'efficientnet_b0' in model_filename:
        architecture_name = 'efficientnet_b0'
    else:
        architecture_name = config['model']['arch']
    
    print(f"[INFO] Architecture: {architecture_name}")
    
    # Load model and data
    config['model']['arch'] = architecture_name
    model = create_model(config)
    model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))
    
    try:
        df = load_and_process_csv(config['csv_path'])
        train_loader, val_loader = get_dataloaders(df, config)
        print(f" Loaded data - Training: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}")
    except Exception as e:
        print(f" Could not load data: {e}")
        return None
    
    # Get original model metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
   
    print("ORIGINAL MODEL EVALUATION\n")
   
    
    model = model.to(device)
    original_metrics = evaluate_pruned_model(model, val_loader, device)
    original_sparsity, _, total_params = calculate_sparsity(model)
    
    
    temp_original_path = "temp_original_pruning.pth"
    torch.save(model.state_dict(), temp_original_path)
    original_size = get_model_size(temp_original_path)
    os.remove(temp_original_path)
    
    print(f"Original Model - Size: {original_size:.1f} MB, Sparsity: {original_sparsity:.4f}, Params: {total_params:,}")
    
    # Apply pruning
    print(f"\n" + "="*60)
    print("APPLYING PRUNING")
    print("="*60)
    
    if pruning_method == 'structured':
        pruned_modules = apply_structured_pruning(model, pruning_ratio,num_classes=config['model']['num_classes'])
    elif pruning_method == 'global':
        pruned_modules = apply_global_magnitude_pruning(model, pruning_ratio)
    else:
        print(f"Unknown pruning method: {pruning_method}")
        return None
    
    # Calculate sparsity after pruning
    post_prune_sparsity, zero_params, total_params = calculate_sparsity(model)
    print(f"After Pruning - Sparsity: {post_prune_sparsity:.4f} ({zero_params:,}/{total_params:,} params)")
    
    # Evaluate immediately after pruning
 
    print("POST-PRUNING EVALUATION (Before Fine-tuning)\n")
   
    
    post_prune_metrics = evaluate_pruned_model(model, val_loader, device)
    
    # Fine-tune if requested
    training_history = None
    if fine_tune_epochs > 0:
        print("FINE-TUNING PRUNED MODEL\n")
       
        
        model, training_history = fine_tune_pruned_model(
            model, train_loader, val_loader, config, device, fine_tune_epochs
        )
    
    # Final evaluation
    
    print("FINAL EVALUATION (After Fine-tuning)\n")
    
    
    final_metrics = evaluate_pruned_model(model, val_loader, device)
    
    # Make pruning permanent
    model = make_pruning_permanent(model)

    slimmed = shrink_classifier_head(model, atol=0.0)
    pruned_save_path = config['pruned_save_path']
    # Save pruned model
    pruned_save_path = pruned_save_path.replace('.pth', f'_pruned_{pruning_method}_{int(pruning_ratio*100)}.pth')
    torch.save(slimmed.state_dict(), pruned_save_path)
    pruned_size = get_model_size(pruned_save_path)

    slimmed = slimmed.to(device).eval()
    final_metrics = evaluate_pruned_model(slimmed, val_loader, device)  
    
    # Create results
    results = {
        'architecture': architecture_name,
        'original_model_path': trained_model_path,
        'pruned_model_path': pruned_save_path,
        'pruning_method': pruning_method,
        'pruning_ratio': pruning_ratio,
        'fine_tune_epochs': fine_tune_epochs,
        'original_size_mb': original_size,
        'pruned_size_mb': pruned_size,
        'size_reduction_percent': ((original_size - pruned_size) / original_size) * 100,
        'original_sparsity': original_sparsity,
        'final_sparsity': post_prune_sparsity,
        'total_parameters': total_params,
        'zero_parameters': zero_params,
        'original_metrics': original_metrics,
        'post_prune_metrics': post_prune_metrics,
        'final_metrics': final_metrics,
        'training_history': training_history,
        'device_used': str(device)
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("PRUNING SUMMARY")
    print(f"{'='*60}")
    print(f"Architecture: {architecture_name}")
    print(f"Pruning Method: {pruning_method}")
    print(f"Pruning Ratio: {pruning_ratio}")
    print(f"Original Size: {original_size:.1f} MB")
    print(f"Pruned Size: {pruned_size:.1f} MB")
    print(f"Size Reduction: {results['size_reduction_percent']:.1f}%")
    print(f"Sparsity: {original_sparsity:.4f} -> {post_prune_sparsity:.4f}")
    
    if original_metrics and final_metrics:
        original_acc = original_metrics['multiclass']['accuracy']
        final_acc = final_metrics['multiclass']['accuracy']
        acc_drop = original_acc - final_acc
        
        print(f"Accuracy: {original_acc:.4f} -> {final_acc:.4f} (drop: {acc_drop:.4f})")
        
        original_f1 = original_metrics['binary']['f1_score']
        final_f1 = final_metrics['binary']['f1_score']
        f1_drop = original_f1 - final_f1
        
        print(f"Binary F1: {original_f1:.4f} -> {final_f1:.4f} (drop: {f1_drop:.4f})")
    
    # Save results
    results_file = pruned_save_path.replace('.pth', '_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Pruned model saved to: {pruned_save_path}")
    
    return results

def main():
    """
    Main function for model pruning
    """
    if len(sys.argv) > 1:
        trained_model_path = sys.argv[1]
        print(f"[INFO] Using provided model path: {trained_model_path}")
    else:
        print("Please provide the path to your trained model:")
        trained_model_path = input("Model path: ").strip()
    
    if not trained_model_path:
        print("No model path provided!")
        sys.exit(1)
    
    # Get pruning parameters
    print("\nPruning method options:")
    print("1. global - Global magnitude pruning (recommended)")
    print("2. structured - Structured pruning (channels/filters)")
    
    method_choice = input("Choose pruning method (default: global): ").strip().lower() or "global"
    if method_choice not in ['global', 'structured']:
        method_choice = 'global'
    
    try:
        pruning_ratio = float(input("Pruning ratio (0.0-0.9, default 0.4): ").strip() or "0.4")
        pruning_ratio = max(0.0, min(0.9, pruning_ratio))
    except ValueError:
        pruning_ratio = 0.4
    
    try:
        fine_tune_epochs = int(input("Fine-tuning epochs (default 3): ").strip() or "3")
    except ValueError:
        fine_tune_epochs = 3
    
    try:
        config = load_config()
        
        results = comprehensive_pruning_pipeline(
            trained_model_path, config, method_choice, pruning_ratio, fine_tune_epochs
        )
        
        if results:
            print("\nModel pruning completed successfully!")
            print(f"Model size reduced by {results['size_reduction_percent']:.1f}%")
            print(f"Sparsity increased to {results['final_sparsity']:.4f}")
        else:
            print("\nModel pruning failed!")
            
    except Exception as e:
        print(f"\nError running model pruning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()