import torch

def calculate_super_aggressive_focal_weights(class_counts_dict):
    weights = []
    for class_name, count in class_counts_dict.items():
        if count < 500: 
            weight = 25.0  
        elif count < 2000:  
            weight = 10.0  
        elif count < 5000:
            weight = 5.0   
        else:  # Large classes
            weight = 1.0   
        weights.append(weight)
        return torch.FloatTensor(weights)
 