import numpy as np

def calculate_ear(landmarks, indices, img_w, img_h):
    """
    Calculate Eye Aspect Ratio (EAR).
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    # Extract coordinates
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))

    p1, p2, p3, p4, p5, p6 = coords

    # Vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    h = np.linalg.norm(p1 - p4)

    if h == 0: return 0
    ear = (v1 + v2) / (2.0 * h)
    return ear

def map_coordinates(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another (Linear Interpolation).
    """
    return np.interp(value, (in_min, in_max), (out_min, out_max))
