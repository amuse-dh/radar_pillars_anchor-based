import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, H*W*A, num_classes)
            targets: (B, H*W*A)
        """
        # Move targets to the same device as predictions
        targets = targets.to(predictions.device)
        
        if predictions.shape[-1] > 1:
            # Multi-class classification
            probs = F.softmax(predictions, dim=-1)  # (B, H*W*num_anchors, num_classes)
            pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, H*W*num_anchors)
        else:
            # Binary classification
            probs = torch.sigmoid(predictions)
            pt = torch.where(targets == 1, probs, 1-probs)
        
        # Compute focal loss
        alpha = torch.ones_like(targets).float() * self.alpha
        alpha = torch.where(targets == 1, alpha, 1-alpha)
        
        focal_weight = (1-pt).pow(self.gamma)
        focal_loss = -alpha * focal_weight * torch.log(pt + 1e-6)
        
        return focal_loss.mean()

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta
    
    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: (B, H, W, num_anchors, 7) box predictions
            targets: (B, H, W, num_anchors, 7) box targets
            mask: (B, H, W, num_anchors) boolean mask for valid boxes
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < self.beta,
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Add dimension for box parameters
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, masks):
        """
        Args:
            predictions: (B, H*W, num_classes)
            targets: (B, H*W)
            masks: (B, H*W)
        """
        # Flatten predictions and targets
        B = predictions.shape[0]
        predictions = predictions.reshape(B, -1, predictions.shape[-1])  # (B, H*W, num_classes)
        targets = targets.reshape(B, -1)  # (B, H*W)
        masks = masks.reshape(B, -1)  # (B, H*W)
        
        # Apply mask
        valid_mask = masks > 0
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calculate loss
        ce_loss = F.cross_entropy(valid_predictions, valid_targets, reduction='mean')
        return ce_loss

class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: (B, N, 7) box predictions
            targets: (B, N, 7) box targets
            mask: (B, N) boolean mask for valid boxes
        Returns:
            loss: scalar tensor
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < 0.11,
                          0.5 * diff ** 2 / 0.11,
                          diff - 0.5 * 0.11)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, N, 1)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()

class RadarPillarsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reg_weight = config.reg_weight
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()
        
        # Grid parameters
        self.x_min = config.x_min
        self.x_max = config.x_max
        self.y_min = config.y_min
        self.y_max = config.y_max
        self.grid_size = config.pillar_x_size * 2
        
    def assign_targets_grid(self, anchors, gt_boxes, gt_labels):
        """
        Grid cell based matching
        Args:
            anchors: (B, H, W, A, 7) - (x,y,z,l,w,h,r)
            gt_boxes: (B, H, W, 7)
            gt_labels: (B, H, W)
        """
        B = anchors.shape[0]
        H = W = 160  # feature map size
        A = 3  # Use only 3 anchors as in cls_preds
        device = anchors.device
        
        # Initialize targets with same shape as input
        cls_targets = torch.zeros((B, H, W, A), dtype=torch.long, device=device)
        reg_targets = torch.zeros((B, H, W, A, 7), device=device)
        pos_mask = torch.zeros((B, H, W, A), dtype=torch.bool, device=device)
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    if gt_labels[b, h, w] > 0:  
                        pos_mask[b, h, w, :] = True
                        cls_targets[b, h, w, :] = gt_labels[b, h, w]
                        
                        # GT box를 각 anchor에 맞게 expand
                        gt_box_expanded = gt_boxes[b, h, w].unsqueeze(0).expand(A, -1)  # (A, 7)
                        current_anchors = anchors[b, h, w]  # (A, 7)
                        
                        # encode_boxes 사용하여 regression targets 계산
                        reg_targets[b, h, w] = self.encode_boxes(
                            gt_box_expanded.unsqueeze(0).unsqueeze(0),  # (1, 1, A, 7)
                            current_anchors.unsqueeze(0).unsqueeze(0)   # (1, 1, A, 7)
                        ).squeeze(0).squeeze(0)  # (A, 7)

        cls_targets = cls_targets.reshape(B, H*W*A)  
        reg_targets = reg_targets.reshape(B, H*W*A, 7)
        pos_mask = pos_mask.reshape(B, H*W*A)
        
        return cls_targets, reg_targets, pos_mask
    
    def encode_boxes(self, gt_boxes, anchors):
        """
        디버깅을 위해 중간값들 출력
        """
        '''
        print("GT boxes shape:", gt_boxes.shape)
        print("GT boxes sample:", gt_boxes[0,0,0])
        print("Anchors shape:", anchors.shape)
        print("Anchors sample:", anchors[0,0,0])
        '''
        
        # nan이 발생할 수 있는 부분들 체크
        dx = (gt_boxes[..., 0] - anchors[..., 0]) / anchors[..., 5]  # l로 나누기
        dy = (gt_boxes[..., 1] - anchors[..., 1]) / anchors[..., 4]  # w로 나누기
        dz = (gt_boxes[..., 2] - anchors[..., 2]) / anchors[..., 3]  # h로 나누기
        
        # log에 0이나 음수가 들어가면 안됨
        dh = torch.log(gt_boxes[..., 3] / anchors[..., 3])
        dw = torch.log(gt_boxes[..., 4] / anchors[..., 4])
        dl = torch.log(gt_boxes[..., 5] / anchors[..., 5])
        dr = torch.zeros_like(dx)
        
        # 중간 결과 출력
        '''
        print("dx range:", dx.min().item(), dx.max().item())
        print("dy range:", dy.min().item(), dy.max().item())
        print("dz range:", dz.min().item(), dz.max().item())
        print("dh range:", dh.min().item(), dh.max().item())
        print("dw range:", dw.min().item(), dw.max().item())
        print("dl range:", dl.min().item(), dl.max().item())
        '''
        
        encoded = torch.stack([dx, dy, dz, dh, dw, dl, dr], dim=-1)
        return encoded
    
    def forward(self, cls_preds, reg_preds, anchors, gt_boxes, gt_labels):
        """
        Args:
            cls_preds: (B, H*W*A, num_classes)
            reg_preds: (B, H*W*A, 7)
            anchors: (B, H, W, A, 7)
            gt_boxes: (B, H, W, 7)
            gt_labels: (B, H, W)
        """
        # Move all inputs to the same device
        device = cls_preds.device
        anchors = anchors.to(device)
        gt_boxes = gt_boxes.to(device)
        gt_labels = gt_labels.to(device)
        
        '''
        print(f"Input shapes:")
        print(f"cls_preds: {cls_preds.shape}")
        print(f"reg_preds: {reg_preds.shape}")
        print(f"anchors: {anchors.shape}")
        print(f"gt_boxes: {gt_boxes.shape}")
        print(f"gt_labels: {gt_labels.shape}")
        '''
        
        # Assign targets
        cls_targets, reg_targets, pos_mask = self.assign_targets_grid(anchors, gt_boxes, gt_labels)
        
        # Get correct dimensions
        B, H, W, A = anchors.shape[:4]
        num_anchors = H * W * A
        
        # Reshape predictions if needed
        if cls_preds.shape[1] != num_anchors:
            cls_preds = cls_preds.reshape(B, H, W, A, -1)
            cls_preds = cls_preds.reshape(B, num_anchors, -1)
        
        if reg_preds.shape[1] != num_anchors:
            reg_preds = reg_preds.reshape(B, H, W, A, 7)
            reg_preds = reg_preds.reshape(B, num_anchors, 7)
        
        '''
        print(f"\nReshaped predictions:")
        print(f"cls_preds: {cls_preds.shape}")
        print(f"reg_preds: {reg_preds.shape}")
        print(f"cls_targets: {cls_targets.shape}")
        print(f"reg_targets: {reg_targets.shape}")
        print(f"pos_mask: {pos_mask.shape}")
        '''
        
        # Classification loss
        cls_loss = self.focal_loss(cls_preds, cls_targets)
        
        # Regression loss
        reg_loss = self.smooth_l1_loss(reg_preds, reg_targets, pos_mask)
        
        # Total loss
        total_loss = cls_loss + self.reg_weight * reg_loss
        
        return total_loss, {
            'total_loss': total_loss,
            'cls_loss': cls_loss.item(),
            'reg_loss': reg_loss.item()
        }
