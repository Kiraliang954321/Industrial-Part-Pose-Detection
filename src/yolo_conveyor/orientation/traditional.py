"""RGB-only dark-part/width-profile baseline. No masks or simulator state inputs."""

import cv2
import numpy as np

from .geometry import prediction


class TraditionalPredictor:
    def __init__(self, *, threshold: int = 35, width_ratio: float = 1.2, min_length: float = 8.0):
        if not 1 <= threshold <= 254 or not 1 <= width_ratio <= 3:
            raise ValueError("invalid contour baseline parameters")
        self.threshold = threshold
        self.width_ratio = width_ratio
        self.min_length = min_length

    def predict(self, image) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = max(15, int(min(image.shape[:2]) * 0.11) | 1)
        # A rectangular kernel permits OpenCV's fast separable morphology path.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        difference = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        mask = (difference > self.threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        results = []
        for contour in contours:
            if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.0004:
                continue
            coords = contour[:, 0, :].astype(float)
            center = coords.mean(axis=0)
            _, vectors = np.linalg.eigh(np.cov((coords - center).T))
            axis = vectors[:, -1]
            perpendicular = np.array([-axis[1], axis[0]])
            along, across = (coords - center) @ axis, (coords - center) @ perpendicular
            lo, hi = float(along.min()), float(along.max())
            length = hi - lo
            thickness = float(across.max() - across.min())
            if length < 16 or length / max(thickness, 1) < 1.4:
                continue
            low_end, high_end = across[along < lo + length * 0.22], across[along > hi - length * 0.22]
            if len(low_end) < 2 or len(high_end) < 2:
                continue
            widths = [float(np.ptp(low_end)), float(np.ptp(high_end))]
            endpoints = [center + axis * lo + perpendicular * np.median(low_end),
                         center + axis * hi + perpendicular * np.median(high_end)]
            head_index = int(widths[1] > widths[0])
            ratio = max(widths) / max(min(widths), 1.0)
            reason = "ambiguous_end_widths" if ratio < self.width_ratio else None
            points = [endpoints[head_index].tolist(), endpoints[1 - head_index].tolist()]
            x, y, w, h = cv2.boundingRect(contour)
            # Heuristic score is not a calibrated probability or visibility estimate.
            score = float(min(0.99, max(0.01, 1 - 1 / max(ratio, 1.01))))
            results.append(prediction([x, y, x + w, y + h], points, score, [score, score],
                                      min_point_conf=0, min_length=self.min_length, reason=reason))
        return sorted(results, key=lambda r: r["confidence"], reverse=True)
