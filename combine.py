import json
import torch
import torchvision
import csv


def combine_predictions(json_files, iou_threshold=0.5, score_threshold=0.1):
    combined_preds = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            preds = json.load(f)
        for pred in preds:
            img_id = pred["image_id"]
            if img_id not in combined_preds:
                combined_preds[img_id] = []
            combined_preds[img_id].append(pred)
    final_preds = []
    for image_id, preds in combined_preds.items():
        preds_by_cat = {}
        for pred in preds:
            cat = pred["category_id"]
            if cat not in preds_by_cat:
                preds_by_cat[cat] = []
            preds_by_cat[cat].append(pred)
        for cat, pred_list in preds_by_cat.items():
            boxes = []
            scores = []
            for pred in pred_list:
                x, y, w, h = pred["bbox"]
                boxes.append([x, y, x + w, y + h])
                scores.append(pred["score"])
            boxes_tensor = torch.tensor(boxes)
            scores_tensor = torch.tensor(scores)
            keep_indices = torchvision.ops.nms(
                boxes_tensor, scores_tensor, iou_threshold=iou_threshold
            )
            for idx in keep_indices:
                selected_pred = pred_list[idx]
                if selected_pred["score"] < score_threshold:
                    continue
                final_preds.append(selected_pred)
    return final_preds


def get_predicted_numbers(predictions):
    """
    Groups predictions by image_id and for each image.
    For each detection, the digit is computed as (category_id - 1).
    The digits are concatenated to form the whole number prediction.
    """
    pred_dict = {}
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in pred_dict:
            pred_dict[img_id] = []
        pred_dict[img_id].append(pred)
    pred_numbers = {}
    for img_id, preds in pred_dict.items():
        preds_sorted = sorted(preds, key=lambda x: x["bbox"][0])
        digits = [str(x["category_id"] - 1) for x in preds_sorted]
        pred_numbers[img_id] = "".join(digits)
    return pred_numbers


def main():
    json_files = [
        "pred_resnet50.json",
        "pred_resnet50_v2.json",
        "pred_mobile.json",
        "pred_focal.json",
    ]
    combined = combine_predictions(
        json_files, iou_threshold=0.5, score_threshold=0.7
    )
    combined_output_file = "combined_pred.json"
    with open(combined_output_file, "w") as f:
        json.dump(combined, f, separators=(",", ":"))
    print(f"Combined predictions saved to {combined_output_file}")
    predicted_numbers = get_predicted_numbers(combined)
    csv_rows = []
    for image_id, whole_number in predicted_numbers.items():
        csv_rows.append({"image_id": image_id, "pred_label": whole_number})
    csv_output_file = "combined_pred.csv"
    with open(csv_output_file, "w", newline="") as csvfile:
        fieldnames = ["image_id", "pred_label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"CSV output saved to {csv_output_file}")


if __name__ == "__main__":
    main()
