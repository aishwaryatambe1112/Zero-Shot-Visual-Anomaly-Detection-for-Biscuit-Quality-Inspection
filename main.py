import cv2
from vision.preprocess import preprocess_frame
from vision.segment import segment_biscuits
from inference.size_defect import check_size_defect
from inference.burnt_detector import check_burnt
from inference.zero_shot_detector import detect_defect

cap = cv2.VideoCapture("data/videos/conveyor.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess_frame(frame)
    biscuits = segment_biscuits(processed, frame)

    for b in biscuits:
        img = b["image"]
        area = b["area"]
        x, y, w, h = b["bbox"]

        size_defect, _ = check_size_defect(area)
        burnt, _ = check_burnt(img)
        label, conf = detect_defect(img)

        if size_defect:
            final_label = "SIZE DEFECT"
        elif burnt:
            final_label = "BURNT"
        else:
            final_label = label.upper()

        color = (0, 255, 0) if "NORMAL" in final_label else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, final_label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Biscuit Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

