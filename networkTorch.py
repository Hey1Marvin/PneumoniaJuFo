import os
import glob
import random
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
import imgaug.augmenters as iaa
from tqdm import tqdm
import math
import random

# Define configuration class
class DetectorConfig:
    NAME = 'pneumonia'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    BACKBONE = 'resnet50'
    NUM_CLASSES = 2  # background + 1 pneumonia class
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1
    STEPS_PER_EPOCH = 200

config = DetectorConfig()

# Data loading functions
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations

class DetectorDataset(Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super(DetectorDataset, self).__init__()
        self.image_fps = image_fps
        self.image_annotations = image_annotations
        self.orig_height = orig_height
        self.orig_width = orig_width

    def __len__(self):
        return len(self.image_fps)

    def __getitem__(self, idx):
        image_fp = self.image_fps[idx]
        annotations = self.image_annotations[image_fp]
        ds = pydicom.read_file(image_fp)
        image = ds.pixel_array
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        mask, class_ids = self.load_mask(annotations)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.uint8)
        class_ids = torch.tensor(class_ids, dtype=torch.int64)
        target = {'boxes': self.get_boxes(mask), 'labels': class_ids, 'masks': mask}
        return image, target

    def load_mask(self, annotations):
        count = len(annotations)
        if count == 0:
            mask = np.zeros((self.orig_height, self.orig_width, 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((self.orig_height, self.orig_width, count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                print("Enumerate I:", i, "  a: ", a)
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(bool), class_ids.astype(np.int32)

    def get_boxes(self, mask):
        boxes = []
        for i in range(mask.shape[2]):
            pos = np.where(mask[:, :, i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.as_tensor(boxes, dtype=torch.float32)

# Set paths and load annotations
ROOT_DIR = ""
train_dicom_dir = "stage_2_train_images"
anns = pd.read_csv(os.path.join(ROOT_DIR, 'stage_2_train_labels.csv'))
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
ORIG_SIZE = 1024

head_size = 2
print("Dataset: ")
print("image_fps: ", image_fps[:head_size])
print("annotations: ", list(image_annotations.items())[:head_size])

pneu_mask = 0
pneu_noMask = 0
ges_mask = 0
ges_noMask = 0
print("first item: ", list(image_annotations.values())[0])
for i in image_annotations.values():
    i = i[0]
    if i['Target'] == 0:
        if i['x'] != i['x']:
            ges_noMask +=1
        else: ges_mask +=1
    else:
        if i['x'] != i['x']: pneu_noMask +=1
        else: pneu_mask += 1

print("Statistic:")
print("pneu_mask: ", pneu_mask)
print("pneu_nomask: ", pneu_noMask)
print("ges_mask: ", ges_mask)
print("ges_nomaks: ", ges_noMask)

# Modify this line to use more or fewer images for training/validation. 
image_fps_list = list(image_fps[:1000])

# Split dataset into training vs. validation dataset
random.seed(42)
random.shuffle(image_fps_list)
validation_split = 0.1
split_index = int((1 - validation_split) * len(image_fps_list))
image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]

print(len(image_fps_train), len(image_fps_val))

# Prepare the training and validation datasets
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)


# DataLoader
data_loader_train = DataLoader(dataset_train, batch_size=config.IMAGES_PER_GPU, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
data_loader_val = DataLoader(dataset_val, batch_size=config.IMAGES_PER_GPU, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Model setup
backbone = resnet_fpn_backbone(config.BACKBONE, pretrained=True)
anchor_generator = AnchorGenerator(
    sizes=tuple((config.RPN_ANCHOR_SCALES,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * len(config.RPN_ANCHOR_SCALES)
)
model = MaskRCNN(backbone, num_classes=config.NUM_CLASSES, rpn_anchor_generator=anchor_generator)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and LR scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.006, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Data augmentation
augmentation = iaa.Sequential([
    iaa.OneOf([
        iaa.Affine(
            scale={"x": (0.98, 1.04), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.03, 0.03), "y": (-0.05, 0.05)},
            rotate=(-5, 5),
            shear=(-3, 3),
        ),
        iaa.PiecewiseAffine(scale=(0.002, 0.03)),
    ]),
    iaa.OneOf([
        iaa.Multiply((0.85, 1.15)),
        iaa.ContrastNormalization((0.85, 1.15)),
    ]),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 0.12)),
        iaa.Sharpen(alpha=(0.0, 0.12)),
    ]),
])

def apply_augmentation(image, target):
    image = image.permute(1, 2, 0).numpy()
    masks = target['masks'].numpy()
    augmented = augmentation(image=image, segmentation_maps=masks)
    image_aug, masks_aug = augmented[0], augmented[1]
    image_aug = torch.tensor(image_aug).permute(2, 0, 1)
    masks_aug = torch.tensor(masks_aug)
    target['masks'] = masks_aug
    return image_aug, target

# Training loop with model saving
if __name__ == "__main__":
    num_epochs = 50
    best_loss = float('inf')
    model_dir = "path/to/save/model"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        i = 0
        with tqdm(total=len(data_loader_train), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, targets in data_loader_train:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                images_aug, targets_aug = [], []
                for image, target in zip(images, targets):
                    image_aug, target_aug = apply_augmentation(image, target)
                    images_aug.append(image_aug)
                    targets_aug.append(target_aug)
                
                loss_dict = model(images_aug, targets_aug)
                losses = sum(loss for losses in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                
                if i % 10 == 0:
                    print(f"Iteration {i}, Loss: {losses.item()}")
                i += 1
                
                pbar.update(1)
        
        lr_scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(data_loader_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
        
        # Save the model if the current epoch's loss is lower than the best loss recorded
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}loss{best_loss:.4f}.pth"))
            print(f"Model saved at epoch {epoch+1} with loss {best_loss:.4f}")

    # Optionally, you can evaluate the model on the validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in data_loader_val:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
        
        avg_val_loss = val_loss / len(data_loader_val)
        print(f"Validation Loss: {avg_val_loss}")