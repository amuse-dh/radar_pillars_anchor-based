# -*- coding: utf-8 -*-

import torch
import numpy as np

class Config:
    model_name = 'radar_pillars'
    
    # Data augmentation
    use_augmentation = True
    random_flip_prob = 0.5
    random_rotate_prob = 0.5
    rotate_range = [-45, 45]  # degrees
    
    # NMS parameters
    nms_iou_threshold = 0.5
    score_threshold = 0.1
    
    criterion = {
        'name': 'RadarPillarsLoss', 
        'params': {
            'reg_weight': 2.0,
            'focal_loss': {
                'alpha': 0.25,
                'gamma': 2.0
            },
            'smooth_l1_loss': {
                'beta': 0.11
            }
        }
    }
    
    def __init__(self):
        self.class_names = ['car', 'truck', 'bus']
        
        self.save_dir = 'checkpoints'
        
        # split the data path
        self.train_point_cloud_path = "/root/datasets/dataset/mobis/train/pcl_filter/"
        self.train_label_path = "/root/datasets/dataset/mobis/train/labels_/"
        self.val_point_cloud_path = "/root/datasets/dataset/mobis/val/pcl_filter/"
        self.val_label_path = "/root/datasets/dataset/mobis/val/labels/"
        
        self.num_input_features = 4  # x, y, z, intensity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Point cloud 
        self.max_points = 20000          
        self.max_pillars = 12000         
        self.max_points_per_pillar = 100 
        self.min_points_in_pillar = 1    
        self.point_feature_dim = 4       
        
        # Point cloud range
        self.x_min, self.x_max = -100.0, 100.0
        self.y_min, self.y_max = -100.0, 100.0
        self.z_min, self.z_max = -0.25, 5.0
        
        # Pillar generate
        self.pillar_x_size = 0.625  # meters
        self.pillar_y_size = 0.625  # meters
        self.num_pillars_x = int((self.x_max - self.x_min) / self.pillar_x_size)
        self.num_pillars_y = int((self.y_max - self.y_min) / self.pillar_y_size)
        self.feature_map_stride = 1.25
        
        
        # Model 
        self.num_classes = 3             
        self.input_dim = 7              
        self.channels = 32              
        self.hidden_dim = 128            
        
        # Training 
        self.batch_size = 8
        self.num_workers = 4
        self.num_epochs = 300
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.pin_memory = True
        
        # Loss 
        self.alpha = 0.25               # focal loss alpha
        self.gamma = 2.0                # focal loss gamma
        self.reg_weight = 2.0           # regression loss weight
        
        # Learning rate scheduler 
        self.initial_lr = 0.0003
        self.max_lr = 0.003
        self.lr_scheduler_factor = 0.1
        self.lr_scheduler_patience = 30
        
        # Early stopping 
        self.early_stopping_patience = 50
        self.early_stopping_min_delta = 1e-4
        
        # Logging 
        self.log_interval = 10          # steps
        self.val_interval = 1           # epochs
        
        # Create save directory if it doesn't exist
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Calculate derived parameters
        self.grid_size = (self.num_pillars_x, self.num_pillars_y)
        self.voxel_size = (self.pillar_x_size, self.pillar_y_size)
        self.grid_range = (
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.z_min, self.z_max
        )
        
        # Layer freezing 
        self.layers_to_freeze = []
        
        # AMP 
        self.use_amp = False
        
        # Random seed 
        self.seed = 42
        
        self.num_anchors = 3
        # Anchor settings
        self.anchor_sizes = [
            [1.5, 1.6, 4.6],  # car (h, w, l)
            [2.5, 2.5, 8.0],  # truck
            [4.0, 2.8, 12.0]  # bus
        ]
        self.anchor_rotations = [0, np.pi/4, -np.pi/4]  # rotation angles
        self.anchor_z_center = -0.6  # default z center
        self.anchor_match_iou_thresh = 0.1  # IoU threshold for positive samples
        self.anchor_match_neg_iou_thresh = 0.4  # IoU threshold for negative samples
        
        # Number of anchors
        self.num_anchor_sizes = len(self.anchor_sizes)
        self.num_anchor_rotations = len(self.anchor_rotations)
        self.num_anchors_per_position = 3
    
    def print_config(self):
        print("\n=== Configuration ===")
        for attr in dir(self):
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                print(f"{attr}: {getattr(self, attr)}")
        print("===================\n")

    def validate_shapes(self, tensor_dict):
        
        B = tensor_dict['pillars'].size(0)
        for name, tensor in tensor_dict.items():
            if name.startswith('pillar'):
                assert tensor.size(0) == B, f"Batch size mismatch in {name}"