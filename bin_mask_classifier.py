import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')

def get_model():
    num_classes = 1
    #Load the model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    #Freeze the convolutional layers
    for param in model.parameters():
        if isinstance(param, nn.Conv2d):
            param.requires_grad = False

    #Modify the models head
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )

    preprocess = weights.transforms()

    return model, preprocess

def get_data():
    images_path = "./data/bin_masks/"
    labels_path = "./data/masks.csv"
    df = pd.read_csv(labels_path)
    names = df["name"].values
    labels = df["label"].values
    images = []
    for name in names:
        image = cv2.imread(os.path.join(images_path, name))
        images.append(image)
    
    return images[:1313], labels[:1313]

def train_model(model, trainLoader, testLoader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.BCELoss()
    model.to(DEVICE)
    model.train()
    
    for epoch in range(10):
        losses_per_epoch = []
        for i, (X, y) in enumerate(trainLoader):
            images = X
            labels = y.unsqueeze(1)
            #print(images.shape)
            images = images.to(DEVICE)
            #print(labels.shape)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            
            loss_value = loss(output, labels)
            loss_value.backward()
            optimizer.step()
            losses_per_epoch.append(loss_value.item())

        print(f"Epoch: {epoch} Loss: {sum(losses_per_epoch) / len(losses_per_epoch)}")
        test_model(model, testLoader)
    return model

def test_model(model, testLoader):
    model.eval()
    correct = 0
    total = 0
    for i, (X, y) in enumerate(testLoader):
        image = X.to(DEVICE)
        label = y.to(DEVICE).unsqueeze(1)
        output = model(image)
        predicted = (output > 0.5).float()
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f"Accuracy: {100 * correct / total}")


#prepocess = weights.transforms()
if __name__ == "__main__":
    model, preprocess = get_model()
    X_all, y_all = get_data()
    print("Total images: ", len(X_all))
    print("Total labels: ", len(y_all))
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)
    print("Train images: ", len(X_train))
    print("Train labels: ", len(y_train))
    print("Validation images: ", len(X_val))
    print("Validation labels: ", len(y_val))
    train_data = [(torch.tensor(X_train[i]).permute(2, 0, 1).float(), torch.tensor(y_train[i]).float()) for i in range(len(X_train))]
    test_data = [(torch.tensor(X_val[i]).permute(2, 0, 1).float(), torch.tensor(y_val[i]).float()) for i in range(len(X_val))]
    trainLoader = DataLoader(train_data, batch_size=32, shuffle=True)
    testLoader = DataLoader(test_data, batch_size=32, shuffle=False)
    model = train_model(model, trainLoader, testLoader)
    torch.save(model.state_dict(), "./models/bin_mask_classifier.pth")
