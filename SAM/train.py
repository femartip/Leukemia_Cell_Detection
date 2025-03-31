from dataset import load_dataset
from torch.optim import Adam
import monai
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel
from tqdm import tqdm
from statistics import mean
import torch
from dataset import SAMDataset

## Create PyTorch DataLoader  
def load_dataset(dataset_name: str, processor: SamProcessor):
  train_dataset = SAMDataset(dataset=dataset_name, processor=processor)

  train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

  batch = next(iter(train_dataloader))
  for k,v in batch.items():
    print(k,v.shape)

  print(batch["ground_truth_mask"].shape)
  return train_dataloader

## Load the model
def load_model():
  model = SamModel.from_pretrained("facebook/sam-vit-base")

  # make sure we only compute gradients for mask decoder
  for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
      param.requires_grad_(False)

  return model

## Train the model
def train_model(model: SamModel, train_dataloader: DataLoader):
  optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

  seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
  num_epochs = 100

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  model.train()
  for epoch in range(num_epochs):
      epoch_losses = []
      for batch in tqdm(train_dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        epoch_losses.append(loss.item())

      print(f'EPOCH: {epoch}')
      print(f'Mean loss: {mean(epoch_losses)}')

  return model

## Export the model
def export_model(model: SamModel, path: str):
  model.save_pretrained(path)

def main():
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  train_dataloader = load_dataset("coco", processor)
  model = load_model()
  model = train_model(model, train_dataloader)
  export_model(model, "sam_model.pt")
