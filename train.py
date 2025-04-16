import os
import json
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# ------------- Data Augmentation and Transforms -------------
class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """
    Converts the PIL image into a torch Tensor.
    """

    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target


class Normalize(object):
    """
    Applies normalization using the provided mean and std.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        return image, target


def get_transform(train):
    """
    Returns a composed transformation.
    """
    # If additional safe augmentations are needed, add them here.
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        return Compose([ToTensor(), normalize])
    else:
        return Compose([ToTensor(), normalize])


# ------------- Dataset -------------
class DigitDataset(Dataset):
    """
    Dataset class for digit images with COCO-style annotations.
    Assumes the annotation file (JSON) contains "images" and "annotations".
    """

    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self.images = data["images"]
        self.annotations = data.get("annotations", [])

        # Map image id to image info.
        self.imgs_dict = {img["id"]: img for img in self.images}

        # Group annotations by image_id.
        self.imgs_annotations = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.imgs_annotations:
                self.imgs_annotations[img_id] = []
            self.imgs_annotations[img_id].append(ann)

        self.ids = list(self.imgs_dict.keys())

    def __getitem__(self, idx):
        # Retrieve image information and load the image.
        img_id = self.ids[idx]
        img_info = self.imgs_dict[img_id]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Retrieve annotations and process bounding boxes.
        anns = self.imgs_annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            # bbox format in annotation: [x, y, w, h]
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

        # Convert boxes to tensor and change format
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y + h
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


# ------------- Training and Evaluation Functions -------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(
        data_loader, desc=f"Epoch {epoch} Training", leave=False
    )
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=losses.item())

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} training loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(
    model, data_loader, device, valid_ann_file, conf_threshold=0.05
):
    """
    Runs inference on the validation set and computes mAP.
    """
    model.eval()
    predictions = []
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                img_id = int(targets[i]["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                # Filter predictions based on confidence score.
                inds = np.where(scores >= conf_threshold)[0]
                boxes = boxes[inds]
                scores = scores[inds]
                labels = labels[inds]
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    pred = {
                        "image_id": img_id,
                        "bbox": [
                            float(x_min),
                            float(y_min),
                            float(w),
                            float(h),
                        ],
                        "score": float(score),
                        "category_id": int(label),
                    }
                    predictions.append(pred)

    # Save predictions temporarily for COCO evaluation.
    temp_pred_file = "temp_predictions.json"
    with open(temp_pred_file, "w") as f:
        json.dump(predictions, f)

    cocoGt = COCO(valid_ann_file)
    cocoDt = cocoGt.loadRes(temp_pred_file)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP = cocoEval.stats[0]  # mAP over IoU=0.50:0.95
    os.remove(temp_pred_file)
    return mAP


# ------------- Main Training Loop -------------
def main():
    # Dataset paths (adjust if necessary)
    train_images_dir = "nycu-hw2-data/train"
    valid_images_dir = "nycu-hw2-data/valid"
    train_ann_file = "nycu-hw2-data/train.json"
    valid_ann_file = "nycu-hw2-data/valid.json"
    # Create datasets with transformations.
    dataset_train = DigitDataset(
        train_images_dir, train_ann_file, transforms=get_transform(train=True)
    )
    dataset_valid = DigitDataset(
        valid_images_dir, valid_ann_file, transforms=get_transform(train=False)
    )

    # DataLoaders with collate function.
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(device)

    # ----- Initialize Model -----
    num_classes = 11  # (10 digits + background)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # ----- Setup Optimizer and Scheduler -----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    num_epochs = 10
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.3
    )

    num_epochs = 10
    early_stop_patience = 5
    best_map = 0.0
    no_improve_epochs = 0
    train_losses = []
    mAP_list = []

    # Training loop.
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        loss = train_one_epoch(
            model, optimizer, data_loader_train, device, epoch
        )
        train_losses.append(loss)
        print("Evaluating on validation set...")
        mAP = evaluate_model(
            model,
            data_loader_valid,
            device,
            valid_ann_file,
            conf_threshold=0.7,
        )
        mAP_list.append(mAP)
        print(f"Epoch {epoch} mAP: {mAP:.4f}")
        lr_scheduler.step()
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        if mAP > best_map:
            best_map = mAP
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print("Early stopping triggered.")
                break

        # Plot training progress after each epoch
        epochs_range = range(1, epoch + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, marker="o", label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs_range,
            mAP_list,
            marker="o",
            color="orange",
            label="Validation mAP",
        )
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title("Validation mAP")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_results.png")
        print("Training results plot saved to training_results.png")


if __name__ == "__main__":
    main()
