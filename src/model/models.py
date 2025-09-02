import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any
from mobileone import mobileone

class ImageClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(ImageClassifier, self).__init__()
        self.config = config
        model_config = config['model']
        self.arch = model_config['arch']
        self.num_classes = model_config['num_classes']
        self.dropout = model_config['dropout']
        self.pretrained = model_config['pretrained']
        # Backbone loader
        self.backbone, self.feature_dim = self._create_backbone(self.arch, self.pretrained)
 
        # Custom classification head
        self.classifier = self._create_classifier_head()
        self._initialize_weights()
 
 
        def _freeze_backbone(self):
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")
 
        def _unfreeze_backbone(self):
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Backbone unfrozen")

        if model_config.get('freeze_backbone', False):
            self._freeze_backbone()

        print(f"Created model: {self.arch}, classes: {self.num_classes}, dropout: {self.dropout}")
 
    def _create_backbone(self, arch: str, pretrained: bool):
        
        if arch == 'mobilenetv3_small':
            backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
 
        elif arch == 'mobilenetv3_large':
            backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
 
        elif arch.startswith('efficientnet_b'):
            backbone_fn = getattr(models,arch)
            backbone = backbone_fn(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn. Identity()
        elif arch == 'mobileone':
            try:
                variant = self.config['model'].get('mobileone_variant','s2')
                backbone = mobileone(num_classes= 1000 ,inference_mode= False, variant=variant )
                feature_dim = backbone.linear.in_features
                backbone.linear = nn.Identity()
            except ImportError:
                backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
                feature_dim = backbone.num_features
 
        else:
            raise ValueError(f"Unsupported model architecture: {arch}")
        return backbone, feature_dim
 
    def _create_classifier_head(self):
        if self.config['model'].get('simple_head',False):
            return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(512, self.num_classes))
        else:
            return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
 
    def _initialize_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
 
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
 
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(features)
 
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)
 
 
def create_model(config: Dict[str, Any]) -> nn.Module:
    return ImageClassifier(config)
 
def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'non_trainable_parameters': total - trainable
    }
 
def get_model_info(model: nn.Module) -> Dict[str, Any]:
    param_info = count_parameters(model)
    return {
        'model_type': model.arch,
        'parameter_info': param_info,
        'model_size_mb': param_info['total_parameters'] * 4 / (1024 * 1024),
        'backbone': model.arch,
        'num_classes': model.num_classes,
        'feature_dim': model.feature_dim
    }