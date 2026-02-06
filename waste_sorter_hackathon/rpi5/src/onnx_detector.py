"""Minimal ONNX Runtime detector for YOLO-style exported models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    """One predicted object."""

    class_id: int
    confidence: float
    box_xyxy: tuple[int, int, int, int]


class ONNXWasteDetector:
    """Run object detection from an ONNX model using CPU on Raspberry Pi."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str],
        imgsz: int = 512,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.class_names = class_names
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError("conf_threshold must be in [0,1]")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0,1]")

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available
        if not providers:
            raise RuntimeError("No ONNX Runtime execution provider available")

        sess_options = ort.SessionOptions()
        if intra_op_threads > 0:
            sess_options.intra_op_num_threads = int(intra_op_threads)
        if inter_op_threads > 0:
            sess_options.inter_op_num_threads = int(inter_op_threads)

        # Keep graph optimization enabled while allowing explicit CPU thread control.
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        in_shape = input_meta.shape
        in_h = in_shape[2] if len(in_shape) > 2 and isinstance(in_shape[2], int) else imgsz
        in_w = in_shape[3] if len(in_shape) > 3 and isinstance(in_shape[3], int) else imgsz

        self.input_height = int(in_h)
        self.input_width = int(in_w)

        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(f"Invalid model input shape: {in_shape}")

        print(f"Loaded ONNX: {self.model_path}")
        print(f"Providers: {self.session.get_providers()}")
        print(f"Input: name={self.input_name}, shape=({self.input_height}, {self.input_width})")
        print(
            "ONNX threads: "
            f"intra_op={sess_options.intra_op_num_threads} "
            f"inter_op={sess_options.inter_op_num_threads}"
        )

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on one BGR frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        blob, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, frame_bgr.shape[:2], scale, pad_x, pad_y)

    def _preprocess(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox + normalize image for model input."""
        src_h, src_w = frame_bgr.shape[:2]

        scale = min(self.input_width / src_w, self.input_height / src_h)
        resized_w = int(round(src_w * scale))
        resized_h = int(round(src_h * scale))

        resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        pad_w = self.input_width - resized_w
        pad_h = self.input_height - resized_h
        pad_x = pad_w / 2.0
        pad_y = pad_h / 2.0

        left = int(np.floor(pad_x))
        right = int(np.ceil(pad_x))
        top = int(np.floor(pad_y))
        bottom = int(np.ceil(pad_y))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(image_chw, axis=0)

        return blob, float(scale), float(left), float(top)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        """Decode model outputs, filter by confidence, and apply NMS."""
        if not outputs:
            return []

        pred = self._standardize_output(outputs[0])
        if pred.size == 0:
            return []

        if pred.shape[1] == 5:
            boxes, scores, class_ids = self._decode_single_class_raw_output(pred)
        elif pred.shape[1] <= 7:
            boxes, scores, class_ids = self._decode_nms_output(pred)
        else:
            boxes, scores, class_ids = self._decode_raw_output(pred)

        if len(scores) == 0:
            return []

        boxes = self._rescale_boxes(
            boxes=boxes,
            original_shape=original_shape,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        detections: list[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(score),
                    box_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def _decode_single_class_raw_output(
        self, pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw single-class output [cx, cy, w, h, score]."""
        if len(self.class_names) != 1:
            raise ValueError(
                "Model output appears to be single-class ([cx,cy,w,h,score]) but runtime "
                f"is configured for {len(self.class_names)} classes. "
                "Use an 8-class ONNX model or align runtime class names."
            )

        raw_boxes = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        class_ids = np.zeros_like(scores, dtype=np.int32)

        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        raw_boxes = raw_boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes = self._xywh_to_xyxy(raw_boxes)
        if boxes.size and np.nanmax(boxes) <= 1.5:
            boxes[:, [0, 2]] *= float(self.input_width)
            boxes[:, [1, 3]] *= float(self.input_height)

        keep_idx = self._nms(boxes, scores, self.iou_threshold)
        if not keep_idx:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        keep_np = np.array(keep_idx, dtype=np.int32)
        return boxes[keep_np], scores[keep_np], class_ids[keep_np]

    def _standardize_output(self, output: np.ndarray) -> np.ndarray:
        """Convert model output into shape [N, K]."""
        arr = np.squeeze(np.asarray(output))

        if arr.ndim == 1:
            return arr.reshape(1, -1)

        if arr.ndim == 2:
            if arr.shape[0] <= 64 and arr.shape[1] > arr.shape[0]:
                return arr.T
            return arr

        if arr.ndim == 3:
            arr = arr[0]
            if arr.shape[0] <= 64 and arr.shape[1] > arr.shape[0]:
                return arr.T
            return arr

        raise ValueError(f"Unsupported ONNX output shape: {output.shape}")

    def _decode_nms_output(
        self, pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode already-NMSed output, expected rows like [x1,y1,x2,y2,conf,cls]."""
        if pred.shape[1] < 6:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        boxes = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        class_ids = pred[:, 5].astype(np.int32)

        if boxes.size and np.nanmax(boxes) <= 1.5:
            boxes[:, [0, 2]] *= float(self.input_width)
            boxes[:, [1, 3]] *= float(self.input_height)

        valid = (scores >= self.conf_threshold) & (class_ids >= 0)
        valid &= class_ids < len(self.class_names)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        return boxes[valid], scores[valid], class_ids[valid]

    def _decode_raw_output(
        self, pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw YOLO output [cx,cy,w,h,class_scores...], then class-aware NMS."""
        if pred.shape[1] < 5:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        raw_boxes = pred[:, :4].astype(np.float32)
        class_scores = pred[:, 4:].astype(np.float32)

        if class_scores.shape[1] != len(self.class_names):
            raise ValueError(
                f"Model output class count {class_scores.shape[1]} does not match "
                f"expected {len(self.class_names)}"
            )

        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        raw_boxes = raw_boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes = self._xywh_to_xyxy(raw_boxes)
        if boxes.size and np.nanmax(boxes) <= 1.5:
            boxes[:, [0, 2]] *= float(self.input_width)
            boxes[:, [1, 3]] *= float(self.input_height)

        keep_indices: list[int] = []
        for class_id in np.unique(class_ids):
            cls_idx = np.where(class_ids == class_id)[0]
            cls_keep = self._nms(boxes[cls_idx], scores[cls_idx], self.iou_threshold)
            keep_indices.extend(cls_idx[i] for i in cls_keep)

        if not keep_indices:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        keep_np = np.array(keep_indices, dtype=np.int32)
        order = np.argsort(scores[keep_np])[::-1]
        keep_np = keep_np[order]

        return boxes[keep_np], scores[keep_np], class_ids[keep_np]

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        """Convert boxes from center xywh to xyxy."""
        boxes = boxes_xywh.copy()
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Basic NMS implementation."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(scores)[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            if order.size == 1:
                break

            rest = order[1:]

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[rest] - inter + 1e-7
            iou = inter / union

            order = rest[np.where(iou <= iou_threshold)[0]]

        return keep

    @staticmethod
    def _rescale_boxes(
        boxes: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> np.ndarray:
        """Map boxes from letterboxed input space back to original frame space."""
        h, w = original_shape
        boxes = boxes.copy()

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale, 1e-8)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale, 1e-8)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

        return boxes
