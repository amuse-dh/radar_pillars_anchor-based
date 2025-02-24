import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarFeatureNet(nn.Module):
    def __init__(self, input_dim, channels):
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        
        # Point-wise feature network
        self.point_net = nn.Sequential(
            nn.Linear(7, channels),  # 4 original + 3 offset = 7 features
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, points, center_coords):
        # points: (batch_size, max_pillars, max_points_per_pillar, 4)
        # center_coords: (batch_size, max_pillars, 3)
        
        batch_size = points.shape[0]
        max_points_per_pillar = points.shape[2]
        
        # Calculate offset from pillar center
        # Expand center_coords to match points shape
        centers = center_coords.unsqueeze(2).expand(-1, -1, max_points_per_pillar, -1)
        
        # Calculate offsets
        offset = points[..., :3] - centers  # (B, P, N, 3)
        
        # Concatenate features: original point (x,y,z,i) + offset (dx,dy,dz)
        point_features = torch.cat([points, offset], dim=-1)  # (B, P, N, 7)
        
        # Reshape for BatchNorm1d
        point_features = point_features.view(-1, point_features.shape[-1])  # (B*P*N, 7)
        
        # Apply point-wise feature network
        transformed = self.point_net(point_features)  # (B*P*N, channels)
        
        # Reshape back
        transformed = transformed.view(batch_size, -1, max_points_per_pillar, self.channels)
        
        # Max pooling across points in each pillar
        pooled = torch.max(transformed, dim=2)[0]  # (B, P, channels)
        
        return pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class PillarAttention(nn.Module):
    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = channels * 4
            
        self.channels = channels
        self.scale = channels ** -0.5  # Attention scaling factor
        
        # QKV projections
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
        # Output projection
        self.out_proj = nn.Linear(channels, channels)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
    
    def forward(self, x, mask):
        B, C, H, W = x.shape
        
        # Reshape features and extract valid pillars
        x_reshaped = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        outputs = []
        
        for b in range(B):
            valid_idx = mask[b].bool()  # (H, W)
            valid_feat = x_reshaped[b][valid_idx]  # (N, C), N = number of valid pillars
            
            if len(valid_feat) == 0:
                attended = torch.zeros((H*W, C), device=x.device, dtype=x.dtype)
            else:
                # Normalization and QKV projections
                valid_feat = self.norm1(valid_feat)
                q = self.q_proj(valid_feat)  # (N, C)
                k = self.k_proj(valid_feat)  # (N, C)
                v = self.v_proj(valid_feat)  # (N, C)
                
                # Calculate attention
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (N, N)
                attn = F.softmax(attn, dim=-1)
                
                # Apply attention
                attended_feat = torch.matmul(attn, v)  # (N, C)
                attended_feat = self.out_proj(attended_feat)
                
                # FFN
                attended_feat = valid_feat + attended_feat  # First residual connection
                attended_feat = self.norm2(attended_feat)
                attended_feat = attended_feat + self.ffn(attended_feat)  # Second residual connection
                
                # Restore spatial information
                attended = torch.zeros((H*W, C), device=x.device, dtype=x.dtype)
                attended[valid_idx.flatten()] = attended_feat
            
            attended = attended.view(H, W, C)
            outputs.append(attended)
        
        output = torch.stack(outputs)  # (B, H, W, C)
        output = output.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return output

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors_per_position):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_position  # 3
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors_per_position * num_classes, kernel_size=1)  # 3 * 3 = 9 channels
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors_per_position * 7, kernel_size=1)  # 3 * 7 = 21 channels
        )
    
    def forward(self, x, grid_size):
        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)
        
        # Classification predictions
        cls_preds = self.cls_head(x)  # (B, num_anchors*num_classes, H, W)
        cls_preds = cls_preds.reshape(batch_size, self.num_anchors, self.num_classes, H, W)
        cls_preds = cls_preds.permute(0, 3, 4, 1, 2)  # (B, H, W, num_anchors, num_classes)
        cls_preds = cls_preds.reshape(batch_size, H*W*self.num_anchors, self.num_classes)  # (B, H*W*num_anchors, num_classes)
        
        # Regression predictions
        reg_preds = self.reg_head(x)  # (B, num_anchors*7, H, W)
        #print(f"reg_preds shape after conv: {reg_preds.shape}")
        reg_preds = reg_preds.reshape(batch_size, self.num_anchors, 7, H, W)
        #print(f"reg_preds shape after first reshape: {reg_preds.shape}")
        reg_preds = reg_preds.permute(0, 3, 4, 1, 2)  # (B, H, W, num_anchors, 7)
        #print(f"reg_preds shape after permute: {reg_preds.shape}")
        reg_preds = reg_preds.reshape(batch_size, H*W*self.num_anchors, 7)  # (B, H*W*num_anchors, 7)
        #print(f"reg_preds final shape: {reg_preds.shape}")
        
        return cls_preds, reg_preds

class AnchorGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.anchor_sizes = config.anchor_sizes[:3]  # 3개의 anchor size만 사용
        self.anchor_rotations = [0]  # rotation은 0만 사용
        self.x_min = config.x_min
        self.x_max = config.x_max
        self.y_min = config.y_min
        self.y_max = config.y_max
        self.z_min = config.z_min
        self.z_max = config.z_max
        self.feature_map_stride = config.feature_map_stride
    
    def forward(self, feature_map_size, batch_size):
        """
        Args:
            feature_map_size: (H, W)
            batch_size: int
        Returns:
            anchors: (B, H, W, 3, 7) tensor of anchors
        """
        # Generate grid coordinates
        x = torch.linspace(self.x_min, self.x_max, feature_map_size[1])
        y = torch.linspace(self.y_min, self.y_max, feature_map_size[0])
        
        y_centers, x_centers = torch.meshgrid(y, x, indexing='ij')
        z_centers = torch.ones_like(x_centers) * self.z_min
        
        # Reshape to (H, W, 1)
        x_centers = x_centers.unsqueeze(-1)
        y_centers = y_centers.unsqueeze(-1)
        z_centers = z_centers.unsqueeze(-1)
        
        # Generate anchors for each position
        anchors = []
        for size in self.anchor_sizes:  # 3개의 size만 사용
            l, w, h = size
            
            # rotation은 0만 사용
            anchor = torch.cat([
                x_centers, y_centers, z_centers,
                torch.ones_like(x_centers) * l,
                torch.ones_like(x_centers) * w,
                torch.ones_like(x_centers) * h,
                torch.zeros_like(x_centers)  # rotation = 0
            ], dim=-1)  # (H, W, 7)
            anchors.append(anchor)
        
        # Stack anchors along new dimension
        anchors = torch.stack(anchors, dim=-2)  # (H, W, 3, 7)
        
        # Expand for batch dimension
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # (B, H, W, 3, 7)
        
        return anchors

class RadarPillars(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config.channels  # Use same number of channels for all stages
        
        # Pillar Feature Net
        self.pillar_feature_net = PillarFeatureNet(input_dim=7, channels=C)
        self.pillar_attention = PillarAttention(C)
        
        # Encoder stages
        self.encoder1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ) for _ in range(2)]  # 3 conv layers
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ) for _ in range(4)]  # 5 conv layers
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ) for _ in range(4)]  # 5 conv layers
        )
        
        # Upsampling layers
        self.up2 = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(C, C, kernel_size=4, stride=4)
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(config)
        
        # Detection head with anchor-based predictions
        total_channels = C * 3  # 3 feature map concatenation
        self.detection_head = DetectionHead(
            in_channels=total_channels,
            num_classes=config.num_classes,
            num_anchors_per_position=config.num_anchors_per_position
        )
    
    def create_pseudo_image(self, pillar_features, center_coords):
        """
        change the pillar features to pseudo image
        Args:
            pillar_features: (B, P, C) tensor
            center_coords: (B, P, 3) tensor containing (x, y, z) coordinates
        Returns:
            pseudo_image: (B, C, H, W) tensor
        """
        batch_size = pillar_features.shape[0]
        pseudo_image = torch.zeros((batch_size, self.config.channels,
                                  self.config.num_pillars_y, self.config.num_pillars_x),
                                 device=pillar_features.device,
                                 dtype=pillar_features.dtype)
        
        point_density = torch.zeros((batch_size, 1,
                               self.config.num_pillars_y,
                               self.config.num_pillars_x),
                              device=pillar_features.device)
        
        # Calculate pillar indices
        x_idxs = ((center_coords[..., 0] - self.config.x_min) / 
                  self.config.pillar_x_size).long()
        y_idxs = ((center_coords[..., 1] - self.config.y_min) / 
                  self.config.pillar_y_size).long()
        
        # Clip indices to valid range
        x_idxs = torch.clamp(x_idxs, 0, self.config.num_pillars_x - 1)
        y_idxs = torch.clamp(y_idxs, 0, self.config.num_pillars_y - 1)
        
        # Scatter pillar features to pseudo-image
        for b in range(batch_size):
            pseudo_image[b, :, y_idxs[b], x_idxs[b]] = pillar_features[b].t()
            point_density[b, 0, y_idxs[b], x_idxs[b]] += 1
        
        point_density = point_density / point_density.max()
        
        return pseudo_image, point_density
    
    def forward(self, points, center_coords, mask):
        # Get pillar features
        pillar_features = self.pillar_feature_net(points, center_coords)
        
        # Convert to pseudo-image
        pseudo_image, point_density = self.create_pseudo_image(pillar_features, center_coords)
        
        
        # Apply attention
        attended = self.pillar_attention(pseudo_image, mask)
        x = pseudo_image + attended
        
        # Encoder stages
        feat1 = self.encoder1(x)        # C, H/2, W/2
        feat2 = self.encoder2(feat1)    # C, H/4, W/4
        feat3 = self.encoder3(feat2)    # C, H/8, W/8
        
        # Upsample feat2 and feat3
        up_feat2 = self.up2(feat2)      # C, H/2, W/2
        up_feat3 = self.up3(feat3)      # C, H/2, W/2
        
        # Calculate target size based on config
        H = self.config.num_pillars_y // 2  # Half size due to first encoder
        W = self.config.num_pillars_x // 2
        target_size = (H, W)
        
        # Ensure all features have exactly the same spatial dimensions
        feat1 = F.interpolate(feat1, size=target_size, mode='bilinear', align_corners=True)
        up_feat2 = F.interpolate(up_feat2, size=target_size, mode='bilinear', align_corners=True)
        up_feat3 = F.interpolate(up_feat3, size=target_size, mode='bilinear', align_corners=True)
        
        # Concatenate features
        multi_scale_features = torch.cat([feat1, up_feat2, up_feat3], dim=1)
        
        # Generate anchors with batch size
        feature_map_size = (self.config.num_pillars_y // 2, self.config.num_pillars_x // 2)
        batch_size = points.size(0)
        anchors = self.anchor_generator(feature_map_size, batch_size)
        
        # Get predictions
        cls_preds, reg_preds = self.detection_head(multi_scale_features, feature_map_size)
        
        #print("anchors.shape: ", anchors.shape)
        
        return cls_preds, reg_preds, anchors

def create_model(config):
    """
    Helper function to create the model with configuration
    """
    model = RadarPillars(config)
    model.anchor_generator = AnchorGenerator(config).to(config.device)
    return model

def train_one_epoch(model, train_loader, optimizer, config):
    for batch_idx, (points, center_coords, mask, cls_targets, reg_targets) in enumerate(pbar):
        assert points.shape[1] == config.max_pillars
        assert points.shape[2] == config.max_points_per_pillar
        assert points.shape[3] == config.num_input_features
