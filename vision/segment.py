import cv2

def segment_biscuits(processed, original):
    _, thresh = cv2.threshold(processed, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    biscuits = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 3000 < area < 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            biscuit_img = original[y:y+h, x:x+w]

            biscuits.append({
                "image": biscuit_img,
                "area": area,
                "bbox": (x, y, w, h)
            })

    return biscuits

