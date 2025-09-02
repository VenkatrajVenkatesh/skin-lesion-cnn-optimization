import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
 
def get_train_transforms(img_size=224):
    
    return A.Compose([
        # Resize the image to a fixed size (img_size x img_size)
        A.Resize(img_size, img_size),

        # Randomly flip the image horizontally
        A.HorizontalFlip(p=0.5),

        # Randomly adjust brightness and contrast 
        A.RandomBrightnessContrast(p=0.5),

        # Randomly rotate the image 
        A.Rotate(limit=20, p=0.5),

        # Random shift, scale, and rotation:
        A.ShiftScaleRotate(
            shift_limit=0.05,      
            scale_limit=0.05,      
            rotate_limit=15,       
            p=0.5
        ),

        # Random changes to hue, saturation, and value (brightness) to simulate lighting variations
        A.HueSaturationValue(
            hue_shift_limit=10,    
            sat_shift_limit=15,    
            val_shift_limit=10,    
            p=0.3
        ),

        # Apply Gaussian blur to simulate out-of-focus or motion blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Normalize the image
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  
            std=(0.229, 0.224, 0.225)  
        ),

        # Convert the image to PyTorch tensor
        ToTensorV2()
    ])
 
def get_valid_transforms(img_size=224):
   
    return A.Compose([
        # Resize the image to a fixed size for model compatibility
        A.Resize(img_size, img_size),

        # Normalize the image 
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)    
        ),

        # Convert image to PyTorch tensor format 
        ToTensorV2()
    ])
def get_test_transforms(config):
   
    img_size = config['data']['img_size']
    return A.Compose([
        # Resize the test image to the required input size
        A.Resize(img_size, img_size),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])
