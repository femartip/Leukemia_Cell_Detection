import torch.optim as optim
import torch.nn as nn
import torch
from cellSAM.sam_inference import CellSAM
from dataloader import BaMboLoader, oursLoader, SegPC21Loader
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import contextlib
from torchmetrics.functional.classification import average_precision
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def calculate_mAP(preds, targets, threshold=None):
    """
    Calculates the mean Average Precision (mAP) for binary segmentation.

    Arguments:
        preds: Tensor of predicted masks (B, H, W).
        targets: Tensor of ground truth masks (B, H, W).
        threshold: Threshold to binarize predictions.

    Returns:
        mean Average Precision (mAP) score.
    """
    #targets = targets
    #if threshold is None:
    #    threshold = [0.5 + i * 0.05 for i in range(10)]

    ap = average_precision(preds, targets, task="binary", thresholds=threshold)
    return ap*100

def finetune_cellsam_decoder(model, train_loader, val_loader, device, num_epochs=500, learning_rate=1e-5):
    max_boxes = 300
    
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False     # Freeze the image encoder

    for param in model.model.prompt_encoder.parameters():
        param.requires_grad = False     # Freeze the prompt encoder

    for name, param in model.model.mask_decoder.named_parameters():
        param.requires_grad = True
        #if 'output_hypernetworks' in name or 'iou_prediction_head' in name:
        #    param.requires_grad = True
        #else:
        #    param.requires_grad = False
    
    print("Model is on device:", next(model.parameters()).device)
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.model.mask_decoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)   
    scaler = GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_ap = 0.0
        running_iou = 0.0
        model.model.mask_decoder.train()

        for batch_idx, (images, target_masks) in enumerate(train_loader):
            images = images.to(device)
            target_masks = target_masks.to(device)
            
            with torch.no_grad():
                # Calls AnchorDETR
                embeddings, paddings = model.generate_embeddings(images, device=device) 
                boxes_per_heatmap = model.generate_bounding_boxes(images, device=device)[0]

            if boxes_per_heatmap.shape[0] > max_boxes:
                indices = torch.randperm(len(boxes_per_heatmap))[:max_boxes]
                boxes_per_heatmap = boxes_per_heatmap[indices]

            optimizer.zero_grad()

            loss = 0
            ap = 0
            mean_iou = 0
            
            #print(f"-- {batch_count+1}/{len(boxes_per_heatmap)}")
            for i, bbox in enumerate(boxes_per_heatmap):        #passes N,4
                #print(f"-- {i+1}/{len(boxes)}")
                while len(bbox.shape) < 2: 
                    bbox = bbox.unsqueeze(0)
                bbox = bbox.to(device)
                
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = model.model.prompt_encoder(
                        points=None,
                        boxes=bbox,
                        masks=None,
                    ) #prompt embedding
                
                with autocast("cuda"):
                    low_res_masks, iou_predictions = model.model.mask_decoder(
                        image_embeddings=embeddings.to(device),
                        image_pe=model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    ) # Forward mask decoder

                    predicted_masks = model.model.postprocess_masks(
                        low_res_masks,
                        input_size=torch.tensor([512, 512]).to(device),
                        original_size=[512, 512],
                    )# Process masks to match target resolution
                
                    predicted_masks = predicted_masks.squeeze(0).float()
            
                    batch_loss = criterion(predicted_masks, target_masks)
                    batch_loss
                    loss += batch_loss

                with torch.no_grad():
                    predicted_probs = torch.sigmoid(predicted_masks)
                    if predicted_probs.max() == 0:
                        logging.warning("Predicted masks are empty")
                    average_precision_score = calculate_mAP(predicted_masks, target_masks.int())
                    ap += average_precision_score
                    mean_iou += iou_predictions[0][0].cpu().detach().numpy()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss)
            
            running_loss += loss.item()/boxes_per_heatmap.shape[0]
            running_ap += ap/boxes_per_heatmap.shape[0]
            running_iou += mean_iou/boxes_per_heatmap.shape[0]

            torch.cuda.empty_cache()
        
        if val_loader is not None:
            model.model.mask_decoder.eval()
            val_loss, val_ap, val_iou = validate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Training Loss: {running_loss/len(train_loader):.4f}, "
              f"Training AP: {running_ap/len(train_loader):.2f}, "
              f"Training IoU: {running_iou/len(train_loader):.4f}",
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation AP: {val_ap:.2f}, "
              f"Validation IoU: {val_iou:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Training Loss: {running_loss/len(train_loader):.4f}, "
              f"Average Precision: {running_ap/len(train_loader):.2f}, "
              f"Mean IoU: {running_iou/len(train_loader):.4f}")
        
    return model


def validate_model(model, val_loader, criterion, device):
    val_loss = 0.0
    with torch.no_grad():
        for images, target_masks in val_loader:
            images = images.to(device)
            target_masks = target_masks.to(device)
            
            embeddings, paddings = model.generate_embeddings(images, device=device)
            boxes_per_heatmap = model.generate_bounding_boxes(images, device=device)[0]
            
            loss = 0
            ap = 0
            mean_iou = 0
            for i, bbox in enumerate(boxes_per_heatmap):
                while len(bbox.shape) < 2: 
                    bbox = bbox.unsqueeze(0)
                bbox = bbox.to(device)
                
                sparse_embeddings, dense_embeddings = model.model.prompt_encoder(
                    points=None,
                    boxes=bbox,
                    masks=None,
                ) #prompt embedding
                
                low_res_masks, iou_predictions = model.model.mask_decoder(
                    image_embeddings=embeddings.to(device),
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                ) # Forward mask decoder

                predicted_masks = model.model.postprocess_masks(
                    low_res_masks,
                    input_size=torch.tensor([512, 512]).to(device),
                    original_size=[512, 512],
                )# Process masks to match target resolution

                predicted_masks = predicted_masks.squeeze(0).float()
                batch_loss = criterion(predicted_masks, target_masks)
                loss += batch_loss

                average_precision_score = calculate_mAP(predicted_masks, target_masks.int())
                ap += average_precision_score
                mean_iou += iou_predictions[0][0].cpu().detach().numpy()
                

            val_loss += loss.item()
    
    return val_loss / len(val_loader), ap / len(val_loader), mean_iou / len(val_loader)


def main():
    #train_data = BaMboLoader(images_dir='./data/BaMbo/train/images', masks_dir='./data/BaMbo/train/masks', image_size=(512, 512))
    #val_data = BaMboLoader(images_dir='./data/BaMbo/validation/images', masks_dir='./data/BaMbo/validation/masks', image_size=(512, 512))
    
    #train_data = oursLoader(images_dir='./data/ours/512_small/images', masks_dir='./data/ours/512_small/masks/img', image_size=(512, 512))

    train_data = SegPC21Loader(images_dir='./data/SegPC21/train/images', masks_dir='./data/SegPC21/train/masks/img', image_size=(512, 512))
    val_data = SegPC21Loader(images_dir='./data/SegPC21/validation/images', masks_dir='./data/SegPC21/validation/masks/img', image_size=(512, 512))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    #val_loader = None

    images, masks = next(iter(train_loader))
    print(images.shape, masks.shape)

    config = {
        'enc_layers': 6,
        'dec_layers': 6,
        'dim_feedforward': 1024,
        'hidden_dim': 256,
        'dropout': 0.0,
        'nheads': 8,
        'num_query_position': 3500,
        'num_query_pattern': 1,
        'spatial_prior': 'learned',
        'attention_type': 'RCDA',
        'num_feature_levels': 1,
        'device': 'cuda',
        'seed': 42,
        'num_classes': 2
    }
    model = CellSAM(config)
    model.load_state_dict(torch.load('./models/cellsam-base.pt'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = model.to(device)
    model = finetune_cellsam_decoder(model, train_loader, val_loader, device, num_epochs=5, learning_rate=0.00001) #1e-5
    torch.save(model.state_dict(), './models/finetuned_bone_marrow_cellsam.pth')

if __name__ == '__main__':
    main()