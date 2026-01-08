import cv2
import csv
from evaluation.metrics import Metrics
from inference.zero_shot_detector import detect_defect
from inference.size_defect import check_size_defect
from inference.burnt_detector import check_burnt

# Toggle modes
MODES = {
    "classical_only": False,
    "vlm_only": False,
    "hybrid": True
}

CLASSES = ["normal", "broken", "burnt", "size defect"]

def run_ablation(mode_name):
    print(f"\nRunning ablation mode: {mode_name}")
    metrics = Metrics(CLASSES)

    with open("data/ground_truth.csv") as f:
        reader = csv.DictReader(f)

        for row in reader:
            img = cv2.imread(f"data/samples/{row['image_name']}")
            gt = row["true_label"]

            if mode_name == "classical_only":
                # Only classical vision logic
                burnt, _ = check_burnt(img)
                size_defect, _ = check_size_defect(img.shape[0] * img.shape[1])

                if size_defect:
                    pred = "size defect"
                elif burnt:
                    pred = "burnt"
                else:
                    pred = "normal"

            elif mode_name == "vlm_only":
                # Only zero-shot VLM
                label, _ = detect_defect(img)
                if "broken" in label:
                    pred = "broken"
                elif "burnt" in label:
                    pred = "burnt"
                elif "size" in label:
                    pred = "size defect"
                else:
                    pred = "normal"

            else:
                # Hybrid (OURS)
                size_defect, _ = check_size_defect(img.shape[0] * img.shape[1])
                burnt, _ = check_burnt(img)
                label, _ = detect_defect(img)

                if size_defect:
                    pred = "size defect"
                elif burnt:
                    pred = "burnt"
                elif "broken" in label:
                    pred = "broken"
                else:
                    pred = "normal"

            metrics.update(gt, pred)

    correct = sum(metrics.matrix[c][c] for c in CLASSES)
    total = sum(sum(metrics.matrix[c].values()) for c in CLASSES)
    accuracy = correct / total

    print(f"Accuracy ({mode_name}): {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    results = {}

    for mode in MODES:
        results[mode] = run_ablation(mode)

    print("\nAblation Summary:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

