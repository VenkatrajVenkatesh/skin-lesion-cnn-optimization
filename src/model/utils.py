import pandas as pd
import os
import ast
import yaml
from sklearn.preprocessing import OneHotEncoder
import csv

 
def load_and_process_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")
 
    
    df['label'] = df.apply(
        lambda row: 'Skin lesion' if row['Type'] == 'Skin lesion'
        else row['OtherLesion'] if row['Type'] == 'Image not useful (skin)'
        else None,
        axis=1
    )
 
    # Keep only required columns
    new_df = df[['ImageFp', 'label', 'Split']].copy()
 
    # Drop rows with missing labels
    new_df = new_df.dropna(subset=['label'])
 
    # Convert string representation of list to actual list
    new_df['label'] = new_df['label'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
    )
 
    # Separate skin lesion and other useful labels
    skin_lesion_df = new_df[new_df['label'] == 'Skin lesion']
    other_useful_df = new_df[
        new_df['label'].apply(lambda x: isinstance(x, list) and len(x) == 1)
    ].copy()
 
    # Flatten list labels to strings
    other_useful_df['label'] = other_useful_df['label'].apply(lambda x: x[0])
 
    # Combine and filter final dataset
    new_df = pd.concat([skin_lesion_df, other_useful_df], ignore_index=True)
 
    allowed_labels = [
        'Skin lesion',
        'image too blurry/ out of focus',
        'not relevant',
        'animal too far away'
    ]
    new_df = new_df[new_df['label'].isin(allowed_labels)]
    label_conversion = {
        'Skin lesion': 'useful',
        'image too blurry/ out of focus': 'blurry',
        'not relevant': 'not relevant',
        'animal too far away': 'too far away'
    }
    new_df['label'] = new_df['label'].map(label_conversion)
 
    config = load_config()
    label_map = config['label_map']
    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, categories=[list(label_map.keys())])
    one_hot_labels = encoder.fit_transform(new_df[['label']])
    one_hot_df = pd.DataFrame(one_hot_labels, columns=label_map.keys())
    new_df = pd.concat([new_df.reset_index(drop=True), one_hot_df], axis=1)
    
 
    return new_df

def load_config(path=# "../config.yaml"):
    
    with open(path, 'r') as file:
        return yaml.safe_load(file)
def get_model_size_mb(model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
        return size_mb