# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import importlib
import logging
import random

import model
from config import Config
from dataset import RadarDataset
from loss import RadarPillarsLoss
from maps import calculate_3d_map_for_folder, post_process_predictions, save_predictions
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_criterion(config):
    
    return RadarPillarsLoss(config)

def freeze_layers(model, layers_to_freeze):
    
    for name, param in model.named_parameters():
        for layer_name in layers_to_freeze:
            if layer_name in name:
                param.requires_grad = False
                logging.info(f"Freezing layer: {name}")

class EarlyStopping:

    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if self.mode == 'min':
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:  
            if val_loss > self.best_loss + self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (points, center_coords, mask, cls_targets, reg_targets, reg_masks, _) in enumerate(progress_bar):
        # Move data to device
        points = points.to(config.device)
        center_coords = center_coords.to(config.device)
        mask = mask.to(config.device)
        cls_targets = cls_targets.to(config.device)
        reg_targets = reg_targets.to(config.device)
        reg_masks = reg_masks.to(config.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        cls_preds, reg_preds, anchors = model(points, center_coords, mask)
        #cls_preds, reg_preds, anchors = model(points, center_coords, mask)
        
        loss, loss_dict = criterion(
            cls_preds,      
            reg_preds,      
            anchors,        
            gt_boxes=reg_targets,    
            gt_labels=cls_targets   
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update statistics
        total_loss += loss_dict['total_loss']
        total_cls_loss += loss_dict['cls_loss']
        total_reg_loss += loss_dict['reg_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'cls_loss': total_cls_loss / (batch_idx + 1),
            'reg_loss': total_reg_loss / (batch_idx + 1),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches
    }

def validate(model, val_loader, criterion, config):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    predictions = []
    progress_bar = tqdm(val_loader, desc=f'validation')
    
    with torch.no_grad():
    
        for batch_idx, (points, center_coords, mask, cls_targets, reg_targets, reg_masks, file_names) in enumerate(progress_bar):
            # Move data to device
            points = points.to(config.device)
            center_coords = center_coords.to(config.device)
            mask = mask.to(config.device)
            gt_boxes = reg_targets.to(config.device)
            gt_labels = cls_targets.to(config.device)
            
            # Forward pass with anchor generation
            cls_preds, reg_preds, anchors = model(points, center_coords, mask)
            
            # Calculate loss
            loss, loss_dict = criterion(
            cls_preds,      
            reg_preds,      
            anchors,        
            gt_boxes=gt_boxes,    
            gt_labels=gt_labels)
            
            #print("reg : ", reg_preds.shape)
            # Update statistics
            total_loss += loss_dict['total_loss']
            total_cls_loss += loss_dict['cls_loss']
            total_reg_loss += loss_dict['reg_loss']
            
            progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'cls_loss': total_cls_loss / (batch_idx + 1),
            'reg_loss': total_reg_loss / (batch_idx + 1)})
            
            # Post-process predictions
            batch_predictions = post_process_predictions(
                cls_preds, reg_preds, anchors,
                score_threshold=config.score_threshold,
                nms_iou_threshold=config.nms_iou_threshold
            )
            predictions.extend(batch_predictions)
            save_predictions(batch_predictions, config.save_dir+"predictions/",file_names)
        
    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches,
        'predictions': predictions
    }

def save_checkpoint(model, optimizer, scheduler, epoch, config, is_best=False):
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }
    
    checkpoint_path = os.path.join(
        config.save_dir,
        f"{config.model_name}_epoch_{epoch}.pth"
    )
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(
            config.save_dir,
            f"{config.model_name}_best.pth"
        )
        torch.save(checkpoint, best_path)

def main():
    # Set random seed
    set_seed(42)  
    
    # Load configuration
    config = Config()
    config.print_config()
    
    # Setup tensorboard
    writer = SummaryWriter(Path(config.save_dir) / 'tensorboard')
    
    # Create model and move to device
    radar_pillars = model.create_model(config)
    radar_pillars = radar_pillars.to(config.device)
    
    # Freeze layers if specified
    if hasattr(config, 'layers_to_freeze'):
        freeze_layers(radar_pillars, config.layers_to_freeze)
    
    # Create datasets and dataloaders
    train_dataset = RadarDataset(config, split='train')
    val_dataset = RadarDataset(config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory, collate_fn=RadarDataset.collate_fn)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory, collate_fn=RadarDataset.collate_fn)
    
    # Learning rate scheduler
    optimizer = torch.optim.Adam(radar_pillars.parameters(), lr=config.initial_lr)
    
    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=config.max_lr/config.initial_lr,
        final_div_factor=1e4
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    )
    
    criterion = get_criterion(config)
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        train_metrics = train_one_epoch(
            radar_pillars, train_loader, criterion, optimizer, scheduler, epoch, config
        )
        
        # Log training metrics
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/Cls_Loss', train_metrics['cls_loss'], epoch)
        writer.add_scalar('Train/Reg_Loss', train_metrics['reg_loss'], epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Validate
        val_metrics = validate(
            radar_pillars, val_loader, criterion, config
        )
        
        # Check for best model
        val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        save_checkpoint(
            radar_pillars, 
            optimizer, 
            scheduler, 
            epoch,
            config,
            is_best=is_best
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Print epoch results
        print(f"Train Loss: {train_metrics['loss']:.4f} (cls: {train_metrics['cls_loss']:.4f}, reg: {train_metrics['reg_loss']:.4f})")
        print(f"Val Loss: {val_metrics['loss']:.4f} (cls: {val_metrics['cls_loss']:.4f}, reg: {val_metrics['reg_loss']:.4f})")
        
        # Calculate mAP
        mAP_3d = calculate_3d_map_for_folder(
            config.val_label_path,
            val_metrics['predictions'],
            iou_threshold=0.5
        )
        print(f"mAP 3D: {mAP_3d:.4f}")
    
    writer.close()

if __name__ == '__main__':
    main()