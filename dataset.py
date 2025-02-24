# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path

class RadarDataset(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config
        self.split = split
        
        if split == "train":
            self.point_cloud_path = Path(config.train_point_cloud_path)
            self.label_path = Path(config.train_label_path)
        else:
            self.point_cloud_path = Path(config.val_point_cloud_path)
            self.label_path = Path(config.val_label_path)
            
        self.point_cloud_files = sorted(list(self.point_cloud_path.glob('*.bin')))
        self.label_files = sorted(list(self.label_path.glob('*.txt')))
        
        
        
        self.label_files = [
            label_file for label_file in self.label_files 
            if (self.point_cloud_path / (label_file.stem + '.bin')).exists()
        ]
        self.point_cloud_files = [
            pc_file for pc_file in self.point_cloud_files 
            if (self.label_path / (pc_file.stem + '.txt')).exists()
        ]
        
        print("pcl len : ", len(self.point_cloud_files))
        print("label len : ", len(self.label_files))
        
        assert len(self.point_cloud_files) == len(self.label_files), \
            "Number of point cloud files and label files must match"
    
    def __len__(self):
        return len(self.point_cloud_files)
    
    @staticmethod
    def collate_fn(batch):      
        try:
            points = []
            center_coords = []
            masks = []
            cls_targets = []
            reg_targets = []
            reg_masks = []
            file_names = []
            
            # 배치 내 최대 pillar 개수 찾기
            max_pillars_in_batch = max(b[0].shape[0] for b in batch)
            
            for sample in batch:
                pillar = sample[0] if torch.is_tensor(sample[0]) else torch.from_numpy(sample[0])
                center_coord = sample[1] if torch.is_tensor(sample[1]) else torch.from_numpy(sample[1])
                
                # 패딩된 pillar 생성
                padded_pillar = torch.zeros((max_pillars_in_batch, pillar.shape[1], pillar.shape[2]), 
                                          dtype=torch.float32)
                padded_pillar[:pillar.shape[0]] = pillar
                points.append(padded_pillar)
                
                # center_coords도 같은 방식으로 패딩
                padded_center = torch.zeros((max_pillars_in_batch, center_coord.shape[1]), 
                                          dtype=torch.float32)
                padded_center[:center_coord.shape[0]] = center_coord
                center_coords.append(padded_center)
                
                masks.append(torch.as_tensor(sample[2]))
                cls_targets.append(torch.as_tensor(sample[3]))
                reg_targets.append(torch.as_tensor(sample[4]))
                reg_masks.append(torch.as_tensor(sample[5]))
                file_names.append(sample[6])
            
            points = torch.stack(points)
            center_coords = torch.stack(center_coords)
            masks = torch.stack(masks)
            cls_targets = torch.stack(cls_targets)
            reg_targets = torch.stack(reg_targets)
            reg_masks = torch.stack(reg_masks)
                
            return points, center_coords, masks, cls_targets, reg_targets, reg_masks, file_names
            
        except Exception as e:
            print(f"Error in collate_fn: {str(e)}")
            print(f"Batch shapes:")
            for i, sample in enumerate(batch):
                print(f"Sample {i}:")
                for j, item in enumerate(sample):
                    if isinstance(item, (torch.Tensor, np.ndarray)):
                        print(f"  Item {j} shape: {item.shape}")
                    else:
                        print(f"  Item {j} type: {type(item)}")
            raise
    
    def read_point_cloud(self, bin_file):
        
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, self.config.point_feature_dim)
        return points
    
    def read_labels(self, label_file):

        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                label = line.strip().split()
                # [class, x, y, z, width, length, height, rotation]
                if label[0] == "car" or label[0] == "Car" or label[0] == "inference_car":
                    label[0] = 0.0
                    
                elif label[0] == "truck" or label[0] == "Truck" or label[0] == "inference_truck":
                    label[0] = 1.0
                    
                else:
                    label[0] = 2.0
                    
                    
                labels.append([
                    float(x) for x in label
                ])
        return np.array(labels)
    
    def preprocess_pointcloud(self, points):

        if len(points) > self.config.max_points:
            choices = np.random.choice(
                len(points), 
                self.config.max_points, 
                replace=False
            )
            points = points[choices]
        
        mask = (points[:, 0] >= self.config.x_min) & (points[:, 0] <= self.config.x_max) & \
               (points[:, 1] >= self.config.y_min) & (points[:, 1] <= self.config.y_max) & \
               (points[:, 2] >= self.config.z_min) & (points[:, 2] <= self.config.z_max)
        points = points[mask]
        
        if points.shape[1] > 3:  
            points[:, 3] = np.clip(points[:, 3], 0, 1)
        
        return points
    
    def create_pillar_features(self, points):

        x_indices = ((points[:, 0] - self.config.x_min) // self.config.pillar_x_size).astype(np.int32)
        y_indices = ((points[:, 1] - self.config.y_min) // self.config.pillar_y_size).astype(np.int32)
        
        valid_indices = (x_indices >= 0) & (x_indices < self.config.num_pillars_x) & \
                       (y_indices >= 0) & (y_indices < self.config.num_pillars_y)
        points = points[valid_indices]
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]
        
        pillar_ids = y_indices * self.config.num_pillars_x + x_indices
        sort_indices = np.argsort(pillar_ids)
        points = points[sort_indices]
        pillar_ids = pillar_ids[sort_indices]
        
        _, counts = np.unique(pillar_ids, return_counts=True)
        
        num_pillars = min(len(counts), self.config.max_pillars)
        pillars = np.zeros((num_pillars, self.config.max_points_per_pillar, points.shape[1]), dtype=np.float32)
        pillar_centers = np.zeros((num_pillars, 3), dtype=np.float32)
        
        start_idx = 0
        valid_pillars = 0
        
        for i, count in enumerate(counts):
            if valid_pillars >= self.config.max_pillars:
                break
                
            num_points = min(count, self.config.max_points_per_pillar)
            if num_points < self.config.min_points_in_pillar:
                continue
                
            pillar_points = points[start_idx:start_idx + num_points]
            pillars[valid_pillars, :num_points] = pillar_points
            
            pillar_centers[valid_pillars] = np.mean(pillar_points[:, :3], axis=0)
            
            valid_pillars += 1
            start_idx += count
        
        mask = np.zeros((self.config.num_pillars_y, self.config.num_pillars_x), dtype=bool)
        mask[y_indices[:valid_pillars], x_indices[:valid_pillars]] = True
        
        pillars = pillars[:valid_pillars]
        pillar_centers = pillar_centers[:valid_pillars]
        
        return pillars, pillar_centers, mask
    
    def create_targets(self, labels):
        """
        Create classification and regression targets
        Args:
            labels: numpy array with shape (N, 8) where each row is
                   [class_id, 0, 0, 0, 0, 0, 0, 0, h,w,l,cx, cy, cz, yaw]
        """
        H = self.config.num_pillars_y // 2  # Half size due to first encoder
        W = self.config.num_pillars_x // 2
        
        cls_targets = torch.zeros((H, W), dtype=torch.long)
        reg_targets = torch.zeros((H, W, 7), dtype=torch.float)
        reg_masks = torch.zeros((H, W), dtype=torch.bool)
        
        for label in labels:
            # label format: [class_id, x, y, z, l, w, h, yaw]
            cls = int(label[0])  # class_id
            x, y = label[11], label[12]  # center coordinates
            
            # Calculate pillar indices
            x_idx = int((x - self.config.x_min) / self.config.pillar_x_size / 2)  # divide by 2 for encoder stride
            y_idx = int((y - self.config.y_min) / self.config.pillar_y_size / 2)
            
            # Ensure indices are within bounds
            if 0 <= x_idx < W and 0 <= y_idx < H:
                # Set classification target
                cls_targets[y_idx, x_idx] = cls
                
                # Set regression target [x, y, z, h, w, l, yaw]
                reg_targets[y_idx, x_idx] = torch.tensor([
                    label[11],  # x
                    label[12],  # y
                    label[13],  # z
                    label[8],  # h
                    label[9],  # w
                    label[10],  # l
                    label[14]   # yaw
                ])
                
                # Set regression mask
                reg_masks[y_idx, x_idx] = True
        
        return cls_targets, reg_targets, reg_masks
    
    def __getitem__(self, index):
        points = self.read_point_cloud(self.point_cloud_files[index])
        labels = self.read_labels(self.label_files[index])
        file_name = os.path.basename(self.point_cloud_files[index]).replace('.bin', '')
        
        points = self.preprocess_pointcloud(points)
        
        pillars, pillar_centers, mask = self.create_pillar_features(points)
        
 
        cls_targets, reg_targets, reg_masks = self.create_targets(labels)
        
        if len(points) == 0:
            points = torch.zeros((self.config.max_pillars, self.config.max_points_per_pillar, 4), dtype=torch.float32)
            center_coords = torch.zeros((self.config.max_pillars, 3), dtype=torch.float32)
        else:
            points = torch.from_numpy(points).float()  # (N, 4)
            center_coords = torch.from_numpy(pillar_centers).float()  # (M, 3)
            
            if len(center_coords.shape) == 1:
                center_coords = center_coords.unsqueeze(0)  # (1, 3)

            if center_coords.shape[0] == 1:
                center_coords = center_coords.repeat(points.shape[0], 1)
            else:
                min_size = min(points.shape[0], center_coords.shape[0])
                points = points[:min_size]
                center_coords = center_coords[:min_size]

            final_points = torch.zeros((self.config.max_pillars, 4), dtype=torch.float32)
            final_centers = torch.zeros((self.config.max_pillars, 3), dtype=torch.float32)

            num_points = min(points.shape[0], self.config.max_pillars)
            final_points[:num_points] = points[:num_points]
            final_centers[:num_points] = center_coords[:num_points]
            
            points = final_points.unsqueeze(1).expand(-1, self.config.max_points_per_pillar, -1)
            center_coords = final_centers
   
        mask = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
        mask = mask.float()
        
        cls_targets = cls_targets if isinstance(cls_targets, torch.Tensor) else torch.from_numpy(cls_targets)
        cls_targets = cls_targets.long()
        
        reg_targets = reg_targets if isinstance(reg_targets, torch.Tensor) else torch.from_numpy(reg_targets)
        reg_targets = reg_targets.float()
        
        reg_masks = reg_masks if isinstance(reg_masks, torch.Tensor) else torch.from_numpy(reg_masks)
        reg_masks = reg_masks.bool()

        assert points.shape == (self.config.max_pillars, self.config.max_points_per_pillar, 4), \
            f"Expected points shape {(self.config.max_pillars, self.config.max_points_per_pillar, 4)}, got {points.shape}"
        assert center_coords.shape == (self.config.max_pillars, 3), \
            f"Expected center_coords shape {(self.config.max_pillars, 3)}, got {center_coords.shape}"
        assert mask.shape == (self.config.num_pillars_y, self.config.num_pillars_x), \
            f"Expected mask shape {(self.config.num_pillars_y, self.config.num_pillars_x)}, got {mask.shape}"
        
        return pillars, pillar_centers, mask, cls_targets, reg_targets, reg_masks, file_name