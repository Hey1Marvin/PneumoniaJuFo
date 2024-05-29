import os
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision
import math
import random
import imgaug
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = pd.read_csv(label_file)
        self.labels.fillna(-1, inplace=True)
        self.labels['Target'] = self.labels['Target'].astype(int)
        self.image_ids = self.labels['patientId'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.dcm')
        dicom = pydicom.read_file(image_path)
        image = dicom.pixel_array
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        image = F.to_tensor(image).float()
        
        boxes = []
        masks = []
        labels = []
        
        patient_data = self.labels[self.labels['patientId'] == image_id]
        for _, row in patient_data.iterrows():
            if row['Target'] == 1:
                boxes.append([row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']])
                mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
                mask[int(row['y']):int(row['y'] + row['height']), int(row['x']):int(row['x'] + row['width'])] = 1
                masks.append(mask)
                labels.append(1)
            else:
                labels.append(0)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return model

def train_model(model, dataloader, device, num_epochs=50, lr=0.005):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, targets in dataloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()
                
                pbar.update(1)
        
        lr_scheduler.step()
        epLoss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epLoss}")
        torch.save(model.state_dict(), os.path.join("", f"model_epoch_{epoch+1}loss{epLoss:.4f}.pth"))
        print(f"Model saved at epoch {epoch+1} with loss {epLoss:.4f}")
    
def evaluate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for output, target in zip(outputs, targets):
                pred_labels = output['labels'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                total += len(true_labels)
                correct += (pred_labels == true_labels).sum()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def test_single_image(model, image_path, device, out_path):
    model.eval()
    dicom = pydicom.read_file(image_path)
    image = dicom.pixel_array
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack([image] * 3, axis=2)
    image_tensor = F.to_tensor(image).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)[0]
    
    masks = output['masks'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    threshold = 0.5  # Adjust as needed
    masks = masks[scores >= threshold]
    
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image, cmap='gray')
    
    for mask in masks:
        mask = mask[0]
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask > 0.5] = [255, 0, 0]  # Red color
        ax.imshow(colored_mask, alpha=0.5)  # Overlay mask with transparency
    
    plt.axis('off')
    plt.show()
    plt.savefig(out_path)



def main():
    image_dir = 'stage_2_train_images'
    label_file = 'stage_2_train_labels.csv'
    train_file = 'train.csv'
    val_file= 'val.csv'
    
    #Datensatz in train und val zerteilen:
    # CSV-Datei einlesen
    df = pd.read_csv(label_file)

    # Zeilen mischen
    df = df.sample(frac=1)

    # Daten in Trainings- und Validierungssets aufteilen
    train_df, val_df = train_test_split(df, test_size=0.08)

    # Daten in CSV-Dateien speichern
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    
    #Trainieren
    dataset = PneumoniaDataset(image_dir, train_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes=2)
    train_model(model, dataloader, device, num_epochs=50)
    
    # Evaluate the model
    print("Evaluating the model...")
    dataset_val = PneumoniaDataset(image_dir, val_file)
    dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    accuracy = evaluate_model(model, dataloader_val, device)
    print(f'Model accuracy: {accuracy:.4f}')
    
    # Test the model on a single image
    image_id = pd.read_csv(val_file)['patientId'][0]
    test_image_path = os.path.join(image_dir, image_id + '.dcm')
    print("Testing the model on a single image...")
    out_path = "test_mrcnn_"+image_id+".png"
    test_single_image(model, test_image_path, device, out_path)

if __name__ == "__main__":
    main()
    
