import torch

@torch.no_grad()
def dice_iou_per_class(logits, targets, num_classes=4):
    predictions = torch.argmax(logits, dim=1)
    results = {}
    for class_id in range(1, num_classes):  # exclude background
        pred = predictions == class_id
        truth = targets == class_id
        intersection = (pred & truth).sum().item()
        pred_sum = pred.sum().item()
        truth_sum = truth.sum().item()
        union = (pred | truth).sum().item()
        dice = (2 * intersection) / (pred_sum + truth_sum + 1e-8)
        iou = intersection / (union + 1e-8)
        results[class_id] = {"dice": dice, "iou": iou}
    return results
