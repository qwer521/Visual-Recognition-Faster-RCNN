import os
import json
import torch
import torchvision.transforms as T
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm


def load_model(
    num_classes=11, model_path="best_model.pth", device="cuda"
):
    """
    Loads the trained Faster R-CNN model with the proper number of classes.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace head to accommodate num_classes (10 digits + background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


class TestDataset(torch.utils.data.Dataset):
    """
    Dataset for test images.
    """

    def __init__(self, test_dir, transforms=None):
        self.test_dir = test_dir
        self.transforms = transforms
        # List image files
        self.image_files = [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        # Sort files numerically based on filename
        self.image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        # Use the numeric part of the filename as image_id
        image_id = int(os.path.splitext(img_file)[0])
        return image, image_id

    def __len__(self):
        return len(self.image_files)


def main():
    # Directories and output filenames
    test_dir = "nycu-hw2-data/test"
    output_pred_json = "pred.json"  # Task 1 output
    output_pred_csv = "pred.csv"  # Task 2 output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(
        num_classes=11, model_path="best_model.pth", device=device
    )

    # Create test dataset and DataLoader with normalization.
    # Use the same normalization as training
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = TestDataset(test_dir, transforms=transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,  # Use custom collate function
    )

    all_predictions = []  # List to hold all detection results (for Task 1)
    # Dictionary to collect detections per image for Task 2.
    predictions_by_image = {}

    # ----- Task 1: Run inference on test images and collect predictions -----
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Test Inference"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, image_id in zip(outputs, image_ids):
                # Retrieve outputs: boxes, scores, labels
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                # Set a confidence threshold (adjust as needed)
                threshold = 0.7
                inds = np.where(scores >= threshold)[0]
                boxes = boxes[inds]
                scores = scores[inds]
                labels = labels[inds]
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    pred = {
                        "image_id": int(image_id),
                        "bbox": [
                            float(x_min),
                            float(y_min),
                            float(w),
                            float(h),
                        ],
                        "score": float(score),
                        "category_id": int(label),
                    }
                    all_predictions.append(pred)

                    # Collect detections per image for Task 2
                    if image_id not in predictions_by_image:
                        predictions_by_image[image_id] = []
                    predictions_by_image[image_id].append(pred)

    # Save Task 1 output to pred.json
    with open(output_pred_json, "w") as f:
        json.dump(all_predictions, f)
    print(f"Task 1: Predictions saved to {output_pred_json}")

    # ----- Task 2: Form whole number from individual digit detections -----
    csv_rows = []
    # Iterate over all test images to ensure every image_id is processed
    for image, image_id in test_dataset:
        if image_id in predictions_by_image:
            # Sort detections by x coordinate (left-to-right)
            dets_sorted = sorted(
                predictions_by_image[image_id], key=lambda d: d["bbox"][0]
            )
            # Map each detection to the corresponding digit.
            # Here, digit = category_id - 1.
            digits = [str(det["category_id"] - 1) for det in dets_sorted]
            whole_number = "".join(digits)
        else:
            whole_number = -1  # No detection found
        csv_rows.append({"image_id": image_id, "pred_label": whole_number})

    # Save Task 2 output to pred.csv
    with open(output_pred_csv, "w", newline="") as csvfile:
        fieldnames = ["image_id", "pred_label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"Task 2: Predictions saved to {output_pred_csv}")


if __name__ == "__main__":
    main()
