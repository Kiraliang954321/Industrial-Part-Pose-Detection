"""Small OpenCV reviewer; geometry and key transitions are usable without a window."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .contract import validate_corners


def display_size(width: int, height: int, max_window: int) -> tuple[int, int]:
    if min(width, height, max_window) <= 0:
        raise ValueError("image dimensions and max window must be positive")
    scale = min(1.0, max_window / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def original_point(x: int, y: int, width: int, height: int,
                   shown_width: int, shown_height: int) -> tuple[float, float]:
    if min(width, height, shown_width, shown_height) <= 0 or not (0 <= x < shown_width and 0 <= y < shown_height):
        raise ValueError("click outside displayed image")
    return x * width / shown_width, y * height / shown_height


def canonical_corners(points: list[tuple[float, float]], width: int, height: int) -> list[list[float]]:
    """Fit an actual rotated rectangle in original pixel space, then normalize."""
    if len(points) != 4 or min(width, height) <= 0:
        raise ValueError("four points and positive dimensions required")
    pixels = np.asarray(points, dtype=np.float32)
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels[:, 0] >= width).any() or (pixels[:, 1] >= height).any():
        raise ValueError("invalid pixel points")
    rect = cv2.minAreaRect(pixels)
    box = cv2.boxPoints(rect)
    corners = [[float(x / width), float(y / height)] for x, y in box]
    validate_corners(corners)  # Includes bounds and strict nonzero convex area.
    return corners


@dataclass
class PointState:
    width: int
    height: int
    shown_width: int
    shown_height: int
    points: list[tuple[float, float]] = field(default_factory=list)

    def click(self, x: int, y: int) -> None:
        if len(self.points) < 4:
            try:
                self.points.append(original_point(x, y, self.width, self.height,
                                                  self.shown_width, self.shown_height))
            except ValueError:
                pass  # Clicks in window chrome/padding are not annotation points.

    def corners(self) -> list[list[float]] | None:
        try:
            return canonical_corners(self.points, self.width, self.height)
        except ValueError:
            return None

    def key(self, key: int) -> str | None:
        key &= 0xff
        if key in (ord("u"), ord("U")):
            if self.points:
                self.points.pop()
        elif key in (ord("r"), ord("R")):
            self.points.clear()
        elif key in (13, 10, 32):
            if self.corners() is not None:
                return "save"
        elif key in (ord("x"), ord("X")):
            return "exclude"
        elif key in (ord("s"), ord("S")):
            return "skip"
        elif key in (ord("q"), ord("Q"), 27):
            return "quit"
        return None


def review_image(image: np.ndarray, *, session_id: str, part_id: str, face: str,
                 class_name: str, index: int, total: int, existing: bool,
                 max_window: int) -> tuple[str, list[list[float]] | None]:
    """Return an operator action, never persist it; closing the window quits safely."""
    height, width = image.shape[:2]
    shown_width, shown_height = display_size(width, height, max_window)
    state = PointState(width, height, shown_width, shown_height)
    displayed = cv2.resize(image, (shown_width, shown_height), interpolation=cv2.INTER_AREA) if (shown_width, shown_height) != (width, height) else image
    window = "OBB annotation"
    header_height = 4 * 25 + 10
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.click(x, y - header_height)

    try:
        cv2.setMouseCallback(window, mouse)
        while True:
            frame = displayed.copy()
            for i, (x, y) in enumerate(state.points):
                px, py = round(x * shown_width / width), round(y * shown_height / height)
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
                if i:
                    prev = state.points[i - 1]
                    cv2.line(frame, (round(prev[0] * shown_width / width), round(prev[1] * shown_height / height)), (px, py), (0, 255, 255), 1)
            corners = state.corners()
            if corners is not None:
                polygon = np.rint(np.asarray(corners) * [shown_width, shown_height]).astype(np.int32)
                cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
            lines = [f"{session_id} / {part_id} | {face} / {class_name}",
                     f"{index}/{total} | existing: {'yes' if existing else 'no'} | points: {len(state.points)}/4" + (" (invalid box)" if len(state.points) == 4 and corners is None else ""),
                     "Left click: add | U: undo | R: reset | Enter/Space: save",
                     "X: exclude | S: skip | Q/Esc: quit"]
            # A separate header avoids hiding pixels underneath the label/controls.
            canvas_width = max(shown_width, 650)  # Keep the controls legible for narrow images.
            header = np.zeros((header_height, canvas_width, 3), dtype=np.uint8)
            for i, line in enumerate(lines):
                cv2.putText(header, line, (5, 22 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1, cv2.LINE_AA)
            canvas = np.zeros((shown_height, canvas_width, 3), dtype=np.uint8)
            canvas[:, :shown_width] = frame
            cv2.imshow(window, np.vstack((header, canvas)))
            key = cv2.waitKeyEx(30)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return "quit", None
            if key != -1:
                action = state.key(key)
                if action:
                    return action, state.corners() if action == "save" else None
    finally:
        cv2.destroyWindow(window)
