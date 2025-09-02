import torch
import torch.nn as nn
from copy import deepcopy

def _nonzero_row_indices(W: torch.Tensor, atol=0.0):
    # rows with any nonzero element
    return (W.abs() > atol).any(dim=1).nonzero(as_tuple=False).flatten()

def _nonzero_col_indices(W: torch.Tensor, atol=0.0):
    # cols with any nonzero element
    return (W.abs() > atol).any(dim=0).nonzero(as_tuple=False).flatten()

def _shrink_linear_pair(lin1: nn.Linear, lin2: nn.Linear, atol=0.0):
    
    W1 = lin1.weight.data.cpu()
    b1 = lin1.bias.data.cpu() if lin1.bias is not None else None

    keep_out = _nonzero_row_indices(W1, atol=atol)
    if keep_out.numel() == 0:
        
        keep_out = torch.tensor([0], dtype=torch.long)

    new_out = keep_out.numel()
    in_features = lin1.in_features

    new_lin1 = nn.Linear(in_features, new_out, bias=(lin1.bias is not None))
    new_lin1.weight.data.copy_(W1[keep_out])
    if b1 is not None:
        new_lin1.bias.data.copy_(b1[keep_out])

    
    W2 = lin2.weight.data.cpu()
    b2 = lin2.bias.data.cpu() if lin2.bias is not None else None

    keep_in = keep_out  
    new_in2 = keep_in.numel()
    new_lin2 = nn.Linear(new_in2, lin2.out_features, bias=(lin2.bias is not None))
    new_lin2.weight.data.copy_(W2[:, keep_in])
    if b2 is not None:
        new_lin2.bias.data.copy_(b2)

    return new_lin1, new_lin2, keep_out

def shrink_classifier_head(model: nn.Module, atol: float = 0.0) -> nn.Module:
   
    if not hasattr(model, "classifier"):
        print("[shrink] No model.classifier found; skipping.")
        return model

    cls = model.classifier
    layers = list(cls.children())
    # find indices of Linear layers
    lin_idx = [i for i, m in enumerate(layers) if isinstance(m, nn.Linear)]
    if len(lin_idx) == 0:
        print("[shrink] No Linear layers in classifier")
        return model

    new_layers = layers[:]  # shallow copy
    # iterate over consecutive Linear pairs
    for i in range(len(lin_idx) - 1):
        idx1, idx2 = lin_idx[i], lin_idx[i + 1]
        lin1: nn.Linear = new_layers[idx1]
        lin2: nn.Linear = new_layers[idx2]
        new_lin1, new_lin2, kept = _shrink_linear_pair(lin1, lin2, atol=atol)
        new_layers[idx1] = new_lin1
        new_layers[idx2] = new_lin2
        
        for j in range(idx1 + 1, idx2):
            if isinstance(new_layers[j], nn.BatchNorm1d):
                # shrink BN channels to match new_lin1.out_features
                bn_old: nn.BatchNorm1d = new_layers[j]
                bn_new = nn.BatchNorm1d(new_lin1.out_features, affine=bn_old.affine, eps=bn_old.eps, momentum=bn_old.momentum)
                if bn_old.affine:
                    bn_new.weight.data.copy_(bn_old.weight.data.cpu()[kept])
                    bn_new.bias.data.copy_(bn_old.bias.data.cpu()[kept])
                bn_new.running_mean.data.copy_(bn_old.running_mean.data.cpu()[kept])
                bn_new.running_var.data.copy_(bn_old.running_var.data.cpu()[kept])
                new_layers[j] = bn_new

    model = deepcopy(model)
    model.classifier = nn.Sequential(*new_layers)
    return model
