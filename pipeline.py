"""YOLOv8n-CRNN pipeline with auditable reading order and batched recognition."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
CRNN_CODE_DIR = APP_DIR
if str(CRNN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CRNN_CODE_DIR))

from ctc_utils import CTCLabelConverter  # noqa: E402
from models import create_model  # noqa: E402
from preprocessing import build_preprocessing  # noqa: E402


@contextlib.contextmanager
def trusted_checkpoint_loading():
    """Allow full loading only while opening trusted local experiment checkpoints."""

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original_load


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Compute intersection over union for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def expand_bbox(bbox, image_width: int, image_height: int, margin_ratio: float):
    """Expand an xyxy box by a fraction of its width and height."""

    x1, y1, x2, y2 = map(float, bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return (
        max(0, int(round(x1 - margin_x))),
        max(0, int(round(y1 - margin_y))),
        min(image_width, int(round(x2 + margin_x))),
        min(image_height, int(round(y2 + margin_y))),
    )


def global_reading_order(items):
    """Return the thesis baseline order: y1 first, then x1."""

    ordered = [dict(item) for item in items]
    ordered.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for order_index, item in enumerate(ordered):
        item["line_id"] = order_index
        item["order_index"] = order_index
        item["order_in_line"] = 0
    return ordered


def _line_geometry(line):
    boxes = [item["bbox"] for item in line]
    centers = [(box[1] + box[3]) / 2.0 for box in boxes]
    heights = [max(1.0, box[3] - box[1]) for box in boxes]
    return median(centers), median(heights), median([box[1] for box in boxes]), median(
        [box[3] for box in boxes]
    )


def line_cluster_reading_order(
    items,
    center_tolerance: float = 0.60,
    min_vertical_overlap: float = 0.20,
):
    """Group boxes into text lines, then order lines and words geometrically."""

    candidates = [dict(item) for item in items]
    candidates.sort(
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2.0,
            item["bbox"][0],
        )
    )
    lines: list[list[dict]] = []

    for item in candidates:
        x1, y1, x2, y2 = item["bbox"]
        center_y = (y1 + y2) / 2.0
        height = max(1.0, y2 - y1)
        compatible = []

        for line_index, line in enumerate(lines):
            line_center, line_height, line_y1, line_y2 = _line_geometry(line)
            center_distance = abs(center_y - line_center)
            overlap = max(0.0, min(y2, line_y2) - max(y1, line_y1))
            overlap_ratio = overlap / max(1.0, min(height, line_height))
            normalized_distance = center_distance / max(height, line_height)
            if (
                normalized_distance <= center_tolerance
                or overlap_ratio >= min_vertical_overlap
            ):
                compatible.append((normalized_distance - overlap_ratio, line_index))

        if compatible:
            _, best_line = min(compatible)
            lines[best_line].append(item)
        else:
            lines.append([item])

    lines.sort(key=lambda line: _line_geometry(line)[0])
    ordered = []
    for line_id, line in enumerate(lines):
        line.sort(key=lambda item: ((item["bbox"][0] + item["bbox"][2]) / 2.0))
        for order_in_line, item in enumerate(line):
            item["line_id"] = line_id
            item["order_in_line"] = order_in_line
            item["order_index"] = len(ordered)
            ordered.append(item)
    return ordered


def baseline_center_reading_order(items, center_gap_ratio: float = 0.60):
    """Cluster words from vertical centers without cross-line overlap shortcuts."""

    candidates = [dict(item) for item in items]
    if not candidates:
        return []
    candidates.sort(
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2.0,
            item["bbox"][0],
        )
    )
    median_height = median(
        max(1.0, item["bbox"][3] - item["bbox"][1]) for item in candidates
    )
    maximum_center_gap = max(4.0, median_height * float(center_gap_ratio))
    lines: list[list[dict]] = []

    for item in candidates:
        center_y = (item["bbox"][1] + item["bbox"][3]) / 2.0
        compatible = []
        for line_index, line in enumerate(lines):
            line_center = median(
                (member["bbox"][1] + member["bbox"][3]) / 2.0
                for member in line
            )
            center_distance = abs(center_y - line_center)
            if center_distance <= maximum_center_gap:
                compatible.append((center_distance, line_index))
        if compatible:
            _, best_line = min(compatible)
            lines[best_line].append(item)
        else:
            lines.append([item])

    lines.sort(
        key=lambda line: median(
            (item["bbox"][1] + item["bbox"][3]) / 2.0 for item in line
        )
    )
    ordered = []
    for line_id, line in enumerate(lines):
        line.sort(key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0)
        for order_in_line, item in enumerate(line):
            item["line_id"] = line_id
            item["order_in_line"] = order_in_line
            item["order_index"] = len(ordered)
            ordered.append(item)
    return ordered


class RevisedOCRPipeline:
    """Load the selected journal models and expose deterministic pipeline stages."""

    def __init__(
        self,
        yolo_weights,
        crnn_weights,
        device="cpu",
        preprocessing_mode="direct_resize",
        imgsz=640,
        nms_iou=0.70,
        max_det=300,
    ):
        self.device = torch.device(device)
        self.imgsz = int(imgsz)
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)
        self.preprocessing_mode = preprocessing_mode

        with trusted_checkpoint_loading():
            self.detector = YOLO(str(yolo_weights))

        checkpoint = torch.load(crnn_weights, map_location=self.device, weights_only=False)
        self.charset = checkpoint["charset"]
        self.converter = CTCLabelConverter(self.charset)
        self.recognizer = create_model(self.charset, pretrained=False, freeze_vgg=True)
        self.recognizer.load_state_dict(checkpoint["model_state_dict"])
        self.recognizer.to(self.device).eval()
        self.transform = build_preprocessing(preprocessing_mode)

    def detect(self, image, confidence=0.001):
        """Run YOLO once and return unsorted xyxy detections."""

        started = time.perf_counter()
        results = self.detector.predict(
            source=image,
            conf=float(confidence),
            iou=self.nms_iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=str(self.device),
            verbose=False,
        )
        detections = []
        for result in results:
            for index in range(len(result.boxes)):
                values = result.boxes.xyxy[index].detach().cpu().tolist()
                detections.append(
                    {
                        "bbox": tuple(float(value) for value in values),
                        "confidence": float(result.boxes.conf[index].detach().cpu()),
                    }
                )
        return detections, 1000.0 * (time.perf_counter() - started)

    @staticmethod
    def crop_detections(image, detections, margin_ratio=0.0):
        """Create BGR crops and retain the expanded coordinates used."""

        image_height, image_width = image.shape[:2]
        crops = []
        for detection in detections:
            crop_bbox = expand_bbox(
                detection["bbox"], image_width, image_height, float(margin_ratio)
            )
            x1, y1, x2, y2 = crop_bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.full((1, 1, 3), 255, dtype=np.uint8)
            crops.append(crop)
        return crops

    @staticmethod
    def _to_pil(crop):
        if isinstance(crop, Image.Image):
            return crop.convert("RGB")
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def recognize_crops(self, crops, batch_size=64):
        """Recognize a crop collection in deterministic batches."""

        outputs = []
        started = time.perf_counter()
        for offset in range(0, len(crops), batch_size):
            batch_crops = crops[offset : offset + batch_size]
            tensors = [self.transform(self._to_pil(crop)) for crop in batch_crops]
            if not tensors:
                continue
            batch = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                logits = self.recognizer(batch)
                probabilities = torch.softmax(logits, dim=2)
                max_probabilities, indices = probabilities.max(dim=2)
            decoded = self.converter.decode_logits(logits)
            for row, text in enumerate(decoded):
                non_blank = indices[row] != 0
                confidence = (
                    float(max_probabilities[row][non_blank].mean().cpu())
                    if bool(non_blank.any())
                    else 0.0
                )
                raw = self.converter.decode_indices(indices[row])
                outputs.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "raw_argmax": " ".join(str(int(value)) for value in indices[row]),
                        "collapsed_argmax": raw,
                    }
                )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        return outputs, elapsed_ms

    def process_image(
        self,
        image,
        confidence=0.25,
        margin_ratio=0.0,
        reading_order="baseline_center_v2",
        center_tolerance=0.60,
        min_vertical_overlap=0.20,
        batch_size=64,
    ):
        """Run the complete detector-recognizer pipeline on one BGR image."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty NumPy array")
        if reading_order not in {
            "baseline_center_v2",
            "line_clustering",
            "global_y1_x1",
        }:
            raise ValueError(f"Unsupported reading order: {reading_order}")

        total_started = time.perf_counter()
        detections, detection_ms = self.detect(image, confidence=confidence)
        if reading_order == "baseline_center_v2":
            ordered = baseline_center_reading_order(
                detections,
                center_gap_ratio=center_tolerance,
            )
        elif reading_order == "line_clustering":
            ordered = line_cluster_reading_order(
                detections,
                center_tolerance=center_tolerance,
                min_vertical_overlap=min_vertical_overlap,
            )
        else:
            ordered = global_reading_order(detections)

        crops = self.crop_detections(image, ordered, margin_ratio=margin_ratio)
        recognition, recognition_ms = self.recognize_crops(crops, batch_size=batch_size)
        results = []
        for detection, crop, prediction in zip(ordered, crops, recognition):
            results.append(
                {
                    "bbox": tuple(int(round(value)) for value in detection["bbox"]),
                    "detection_confidence": detection["confidence"],
                    "line_id": detection["line_id"],
                    "order_in_line": detection["order_in_line"],
                    "order_index": detection["order_index"],
                    "text": prediction["text"],
                    "recognition_confidence": prediction["confidence"],
                    "raw_argmax": prediction["raw_argmax"],
                    "collapsed_argmax": prediction["collapsed_argmax"],
                    "crop": crop,
                }
            )

        lines = {}
        for row in results:
            lines.setdefault(row["line_id"], []).append(row["text"])
        text_lines = [" ".join(lines[line_id]) for line_id in sorted(lines)]
        return {
            "image_shape": tuple(image.shape),
            "num_detections": len(results),
            "results": results,
            "text_lines": text_lines,
            "full_text": "\n".join(text_lines),
            "configuration": {
                "confidence_threshold": float(confidence),
                "crop_margin": float(margin_ratio),
                "reading_order": reading_order,
                "preprocessing": self.preprocessing_mode,
            },
            "timing": {
                "detection_ms": detection_ms,
                "recognition_ms": recognition_ms,
                "total_ms": 1000.0 * (time.perf_counter() - total_started),
            },
        }
