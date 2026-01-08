EXPECTED_AREA = 8500
TOLERANCE = 0.20

def check_size_defect(area):
    lower = EXPECTED_AREA * (1 - TOLERANCE)
    upper = EXPECTED_AREA * (1 + TOLERANCE)

    if area < lower:
        return True, "undersized"
    elif area > upper:
        return True, "oversized"
    else:
        return False, "normal"

