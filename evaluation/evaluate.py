import cv2
import csv
import time
from evaluation.metrics import Metrics
from inference.zero_shot_detector import detect_defect
from inference.size_defect import check_size_defect
from inference.burnt_detector import check_burnt

CLASSES = ["normal", "broken", "burnt", "size defect"]
metrics = Metrics(CLASSES)

total_time = 0
count = 0

with open("data/ground_truth.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img = cv2.imread(f"data/samples/{row['image_name']}")
        gt = row["true_label"]

        start = time.time()

        size_defect, _ = check_size_defect(img.shape[0] * img.shape[1])
        burnt, _ = check_burnt(img)
        vlm_label, _ = detect_defect(img)

        if size_defect:
            pred = "size defect"
        elif burnt:
            pred = "burnt"
        elif "broken" in vlm_label:
            pred = "broken"
        else:
            pred = "normal"

        metrics.update(gt, pred)
        total_time += time.time() - start
        count += 1

print("Average inference time:", total_time / count)
print("FPS:", count / total_time)

for cls in CLASSES:
    print(cls,
          "P:", round(metrics.precision(cls), 3),
          "R:", round(metrics.recall(cls), 3),
          "F1:", round(metrics.f1(cls), 3))

