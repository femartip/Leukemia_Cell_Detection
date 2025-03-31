import logging
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from torchmetrics.functional.classification import average_precision, multilabel_accuracy
from unet import UNet
from pycocotools.coco import COCO
import json

from dataloader import ImageDataLoader

SEED = 42

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.verbose = verbose

    def __call__(self, val_mAP, model):
        score = val_mAP
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"Validation mAP improved to {score:.4f}. Saving model...")
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Validation mAP did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Validation mAP improved to {score:.4f}. Saving model...")

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        if self.verbose:
            print("Loaded best model state.")

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), targets

def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: torch.Tensor, target: torch.Tensor):
    return 1 - dice_coeff(input, target)

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
    if threshold is None:
        threshold = [0.5 + i * 0.05 for i in range(10)]

    ap = average_precision(preds, targets, task="multilabel", thresholds=threshold, num_labels=NUM_CLASSES)
    return ap*100


def evaluate_model(model, val_loader, loss_fn):
    model.eval()
    aps = []
    aps_five = []
    aps_sevenfive = []
    accuracies = []
    losses = []

    with torch.no_grad():
        for images, true_masks in val_loader:
            images = images.to(device)
            true_masks = torch.stack(true_masks).long().to(device)

            masks_pred = model(images).squeeze(1)
            masks_pred = torch.sigmoid(masks_pred)
            
            ap = calculate_mAP(masks_pred, true_masks)
            ap_five = calculate_mAP(masks_pred, true_masks, threshold=[0.5])
            ap_sevenfive = calculate_mAP(masks_pred, true_masks, threshold=[0.75])
            #accuracy_metric.update(masks_pred, true_masks)
            #accuracy = accuracy_metric.compute()
            loss = loss_fn(masks_pred, true_masks.float()) + dice_loss(masks_pred, true_masks.float())

            aps.append(ap)
            aps_five.append(ap_five)
            aps_sevenfive.append(ap_sevenfive)
            #accuracies.append(accuracy)
            losses.append(loss.item())

    mAP = sum(aps) / len(aps)
    mAP_five = sum(aps_five) / len(aps_five)
    mAP_sevenfive = sum(aps_sevenfive) / len(aps_sevenfive)
    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_loss = sum(losses) / len(losses)
    return mAP, mAP_five, mAP_sevenfive, mean_accuracy, mean_loss

def train_model(model,device, train_dataset, val_dataset, epochs=100, learning_rate=0.001, batch_size=4):
    weight_decay = 0.000000001
    #momentum = 0.999
    #gradient_clipping = 1.0

    logging.info(f'''Starting training: Epochs:{epochs} Batch size:{batch_size} Learning rate:{learning_rate} Device:{device.type}''')

    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, verbose=True) 
    early_stopping = EarlyStopping(patience=5, verbose=True)
    loss_ce = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    metrics = {}

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        model.to(device=device)
        model.train(True)
        num_batches = 0
        losses = []
        accuracies = []
        aps = []
        aps_five = []
        aps_sevenfive = []

        for idx, batch in enumerate(train_loader):
            images, true_masks = batch
            images = images.to(device=device)
            true_masks = true_masks.to(device=device)
            logging.debug(f"Batch {idx}, images: {images.shape}, masks: {true_masks.shape}")

            masks_pred = model(images)  # Forward pass
            logging.debug(f"predicted masks: {masks_pred.shape}")
            logging.debug(f"Max value pred: {torch.sigmoid(masks_pred).max()}, Max value true {true_masks.max()}")
            #loss = combined_loss(masks_pred, true_masks, metrics)    # Calculates the loss as combination of BCE and Dice loss, this ensures pixel level precision
            loss = loss_ce(masks_pred, true_masks) 
            loss += dice_loss(torch.sigmoid(masks_pred), true_masks)
            optimizer.zero_grad()   # Zero gradients
            
            loss.backward() 

            optimizer.step()

            logging.debug(f"Loss: {loss.item()}")

            #accuracy = multilabel_accuracy(torch.sigmoid(masks_pred), true_masks, threshold=0.5, num_labels=NUM_CLASSES, average="micro", ignore_index= 4)  # High class imbalance in bone (4 class)
            accuracy = 0
            logging.debug(f"Accuracy: {accuracy}")
            accuracies.append(accuracy)

            try:
                ap = calculate_mAP(torch.sigmoid(masks_pred), true_masks.long())
                ap_five = calculate_mAP(torch.sigmoid(masks_pred), true_masks.long(), threshold=[0.5])
                ap_sevenfive = calculate_mAP(torch.sigmoid(masks_pred), true_masks.long(), threshold=[0.75])
            except Exception as e:
                logging.error(f"Error calculating mAP: {e}")
                logging.error(f"Predictions: {torch.min(masks_pred)}, {torch.max(masks_pred)}")
                logging.error(f"True masks: {torch.min(true_masks)}, {torch.max(true_masks)}")
                ap = 0
            if torch.isnan(ap):
                ap = 0
            if torch.isnan(ap_five):
                ap_five = 0
            if torch.isnan(ap_sevenfive):
                ap_sevenfive = 0

            logging.debug(f"mAP: {ap}")
            aps.append(ap)
            aps_five.append(ap_five)
            aps_sevenfive.append(ap_sevenfive)
            losses.append(loss.item())
            
            num_batches += 1
            if num_batches % 50 == 0:
                print(f"Batch: {num_batches}, Loss: {sum(losses) / num_batches:.4f}, Accuracy: {sum(accuracies) / num_batches:.4f}, mAP: {sum(aps) / num_batches:.6f}, mAP@0.5: {sum(aps_five) / num_batches:.6f}, mAP@0.75: {sum(aps_sevenfive) / num_batches:.6f}")
    
        avg_epoch_loss = sum(losses) / num_batches
        avg_epoch_acc = sum(accuracies) / num_batches
        mean_ap = sum(aps) / num_batches

        print(f"Training Loss: {avg_epoch_loss:.4f}, Training Accuracy: {avg_epoch_acc:.4f}, Training mAP: {mean_ap:.2f}, Training mAP@0.5: {sum(aps_five) / num_batches:.2f}, Training mAP@0.75: {sum(aps_sevenfive) / num_batches:.2f}")
        
        metrics[epoch+1] = {"train_loss": float(avg_epoch_loss), "train_accuracy": float(avg_epoch_acc), "train_mAP": float(mean_ap), "train_mAP_five": float(sum(aps_five) / num_batches), "train_mAP_sevenfive": float(sum(aps_sevenfive) / num_batches)}
        
        if val_dataset is not None:
            mAP, mAP_five, mAP_sevenfive, mean_accuracy, val_loss = evaluate_model(model, val_loader, loss_ce)
            
            metrics[epoch+1]["val_mAP"] = float(mAP)
            metrics[epoch+1]["val_mAP_five"] = float(mAP_five)
            metrics[epoch+1]["val_mAP_sevenfive"] = float(mAP_sevenfive)
            metrics[epoch+1]["val_accuracy"] = (mean_accuracy)
            metrics[epoch+1]["val_loss"] = val_loss

            scheduler.step(mAP)

            print(f"Validation mAP |IoU 0.5:0.95|: {mAP:.2f}, mAP |IoU 0.5|: {mAP_five:.2f}, mAP |IoU 0.75|: {mAP_sevenfive:.2f}, Accuracy: {mean_accuracy:.4f}")

            early_stopping(float(mAP), model)

            if early_stopping.early_stop:
                logging.warning("Early stopping")
                model.load_state_dict(early_stopping.best_model_state)
                mAP, mAP_five, mAP_sevenfive, mean_accuracy, val_loss = evaluate_model(model, val_loader, loss_ce)
                print(f"Final validation mAP |IoU 0.5:0.95|: {mAP:.2f}, mAP |IoU 0.5|: {mAP_five:.2f}, mAP |IoU 0.75|: {mAP_sevenfive:.2f}, Accuracy: {mean_accuracy:.4f}")
                break

    return metrics


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="./data/ours/1024/")
    args.add_argument("--device", type=str, default="cuda:1")
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--learning_rate", type=float, default=0.001)

    args = args.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_dir = args.data_dir
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Using device: ", device)
    if device.type != 'cpu':
        torch.cuda.empty_cache()
    
    #img_shape = (1024, 1024)
    img_shape = (512, 512)
    train_dataset = ImageDataLoader(images_dir=os.path.join(data_dir, "images"), masks_dir=os.path.join(data_dir, "masks"), image_size=img_shape, transform=None)
    test_img, test_mask = train_dataset[0]

    global NUM_CLASSES
    NUM_CLASSES = test_mask.shape[0]
    print(test_img.shape, test_mask.shape)
    model = UNet(num_classes=test_mask.shape[0], img_shape=img_shape)

    result = train_model(model, device, train_dataset, None, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)
    