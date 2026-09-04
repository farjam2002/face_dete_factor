import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


logger = logging.getLogger("attendance")


@dataclass
class DetectedFace:
    """چهره تشخیص داده‌شده در یک فریم"""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    quality_score: float
    is_usable: bool


class FaceDetector:
    """
    تشخیص چهره با ONNX Runtime (UltraFace RFB-320).
    فرمت خروجی: scores (1, 4420, 2) + boxes (1, 4420, 4) نرمال‌شده
    """
    
    def __init__(
        self,
        model_path: str = "models/ultraface-RFB-320.onnx",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        min_face_size: int = 20,
        blur_threshold: float = 50.0,
        brightness_min: int = 30,
        brightness_max: int = 230
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        
        self._mode = "none"
        self._session = None
        self._input_w = 320
        self._input_h = 240
        
        if not ORT_AVAILABLE:
            logger.error("ONNX Runtime not installed")
            self._mode = "skin"
            return
        
        model_file = self._find_file(model_path)
        
        if not model_file or not model_file.exists():
            logger.error(f"Model not found: {model_path}")
            self._mode = "skin"
            return
        
        if model_file.stat().st_size < 100_000:
            logger.error(f"Model file too small: {model_file.stat().st_size} bytes")
            self._mode = "skin"
            return
        
        try:
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.log_severity_level = 3
            
            self._session = ort.InferenceSession(
                str(model_file),
                sess_options=sess_options,
                providers=providers
            )
            
            self._input_name = self._session.get_inputs()[0].name
            input_shape = self._session.get_inputs()[0].shape
            
            if len(input_shape) == 4:
                self._input_h = input_shape[2] if input_shape[2] else 240
                self._input_w = input_shape[3] if input_shape[3] else 320
            
            self._output_names = [o.name for o in self._session.get_outputs()]
            
            self._mode = "ultraface"
            logger.info(
                f"FaceDetector initialized with ONNX Runtime (UltraFace) "
                f"(model: {model_file.name}, input: {self._input_w}x{self._input_h})"
            )
        
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._mode = "skin"
    
    def _find_file(self, file_path: str) -> Optional[Path]:
        p = Path(file_path)
        if not p.is_absolute():
            candidates = [
                p,
                Path(__file__).parent.parent / p,
            ]
            for c in candidates:
                if c.exists():
                    return c
        elif p.exists():
            return p
        return None
    
    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        if frame is None:
            return []
        
        if self._mode == "ultraface":
            return self._detect_ultraface(frame)
        elif self._mode == "skin":
            return self._detect_skin(frame)
        return []
    
    def _detect_ultraface(self, frame: np.ndarray) -> List[DetectedFace]:
        """تشخیص با UltraFace RFB-320 - فرمت دقیق"""
        try:
            h, w = frame.shape[:2]
            
            input_w = self._input_w
            input_h = self._input_h
            
            # Resize
            resized = cv2.resize(frame, (input_w, input_h))
            
            # نرمال‌سازی: (x - 127) / 128
            blob = resized.astype(np.float32)
            blob = (blob - 127.0) / 128.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
            
            # Inference
            results = self._session.run(
                self._output_names,
                {self._input_name: blob}
            )
            
            # خروجی‌ها: scores (1, 4420, 2), boxes (1, 4420, 4)
            if len(results) < 2:
                logger.warning("Unexpected output format")
                return []
            
            # پیدا کردن index صحیح
            scores_idx = None
            boxes_idx = None
            
            for i, result in enumerate(results):
                if result.shape[-1] == 2:
                    scores_idx = i
                elif result.shape[-1] == 4:
                    boxes_idx = i
            
            if scores_idx is None or boxes_idx is None:
                logger.warning("Could not identify scores and boxes")
                return []
            
            scores = results[scores_idx][0]  # (4420, 2)
            boxes = results[boxes_idx][0]    # (4420, 4)
            
            # استخراج face scores (ستون 1)
            face_scores = scores[:, 1]
            
            # فیلتر بر اساس آستانه
            high_conf_indices = np.where(face_scores >= self.score_threshold)[0]
            
            if len(high_conf_indices) == 0:
                return []
            
            # مرتب‌سازی بر اساس confidence
            sorted_indices = high_conf_indices[
                np.argsort(-face_scores[high_conf_indices])
            ]
            
            # استخراج چهره‌ها
            detections = []
            for idx in sorted_indices:
                confidence = float(face_scores[idx])
                x1, y1, x2, y2 = boxes[idx]
                
                # تبدیل از مختصات نرمال‌شده (0-1) به پیکسل
                x1_px = int(x1 * w)
                y1_px = int(y1 * h)
                x2_px = int(x2 * w)
                y2_px = int(y2 * h)
                
                # اطمینان از ترتیب صحیح
                if x1_px > x2_px:
                    x1_px, x2_px = x2_px, x1_px
                if y1_px > y2_px:
                    y1_px, y2_px = y2_px, y1_px
                
                # بررسی محدوده
                x1_px = max(0, min(x1_px, w - 1))
                y1_px = max(0, min(y1_px, h - 1))
                x2_px = max(0, min(x2_px, w - 1))
                y2_px = max(0, min(y2_px, h - 1))
                
                fw = x2_px - x1_px
                fh = y2_px - y1_px
                
                if fw < self.min_face_size or fh < self.min_face_size:
                    continue
                
                # ارزیابی کیفیت
                quality, is_usable = self._evaluate_quality(frame, x1_px, y1_px, fw, fh)
                
                detections.append(DetectedFace(
                    x=x1_px, y=y1_px, w=fw, h=fh,
                    confidence=confidence,
                    quality_score=quality,
                    is_usable=is_usable
                ))
            
            # NMS
            if detections:
                detections = self._nms(detections)
            
            return detections
        
        except Exception as e:
            logger.error(f"UltraFace detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _nms(self, detections: List[DetectedFace]) -> List[DetectedFace]:
        if not detections:
            return []
        
        boxes = []
        scores = []
        
        for det in detections:
            boxes.append([det.x, det.y, det.x + det.w, det.y + det.h])
            scores.append(det.confidence)
        
        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(
            boxes_arr.tolist(),
            scores_arr.tolist(),
            self.score_threshold,
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        indices = indices.flatten()
        return [detections[i] for i in indices]
    
    def _detect_skin(self, frame: np.ndarray) -> List[DetectedFace]:
        try:
            h, w = frame.shape[:2]
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            mask = cv2.inRange(
                ycrcb,
                np.array([0, 77, 133]),
                np.array([255, 127, 173])
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            result = []
            for cnt in contours:
                x, y, fw, fh = cv2.boundingRect(cnt)
                
                if fw < self.min_face_size or fh < self.min_face_size:
                    continue
                
                aspect = fw / fh
                if not (0.6 <= aspect <= 1.4):
                    continue
                
                roi_mask = mask[y:y+fh, x:x+fw]
                skin_ratio = cv2.countNonZero(roi_mask) / (fw * fh)
                if skin_ratio < 0.3:
                    continue
                
                confidence = min(skin_ratio * 2, 0.95)
                quality, is_usable = self._evaluate_quality(frame, x, y, fw, fh)
                
                result.append(DetectedFace(
                    x=x, y=y, w=fw, h=fh,
                    confidence=confidence,
                    quality_score=quality,
                    is_usable=is_usable
                ))
            
            result.sort(key=lambda f: f.confidence, reverse=True)
            return result
        
        except Exception as e:
            logger.error(f"Skin detection error: {e}")
            return []
    
    def _evaluate_quality(
        self, frame: np.ndarray,
        x: int, y: int, w: int, h: int
    ) -> Tuple[float, bool]:
        try:
            y1 = max(0, y)
            y2 = min(frame.shape[0], y + h)
            x1 = max(0, x)
            x2 = min(frame.shape[1], x + w)
            
            if y2 <= y1 or x2 <= x1:
                return 0.0, False
            
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                return 0.0, False
            
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            blur = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            blur_score = min(blur / self.blur_threshold, 1.0)
            
            brightness = np.mean(gray_roi)
            if brightness < self.brightness_min or brightness > self.brightness_max:
                brightness_score = 0.3
            else:
                brightness_score = 1.0
            
            size_score = min(w * h / (150 * 150), 1.0)
            
            quality = (blur_score * 0.5) + (brightness_score * 0.25) + (size_score * 0.25)
            
            is_usable = (
                w >= self.min_face_size and
                h >= self.min_face_size and
                blur_score > 0.05
            )
            
            return quality, is_usable
        
        except Exception:
            return 0.0, False
    
    def draw_faces(
        self,
        frame: np.ndarray,
        faces: List[DetectedFace],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        result = frame.copy()
        
        for i, face in enumerate(faces):
            if face.is_usable and face.confidence > 0.7:
                color = (0, 255, 0)  # سبز
            elif face.is_usable and face.confidence > 0.5:
                color = (0, 255, 255)  # زرد
            elif face.is_usable:
                color = (0, 165, 255)  # نارنجی
            else:
                color = (255, 0, 0)  # آبی
            
            cv2.rectangle(
                result,
                (face.x, face.y),
                (face.x + face.w, face.y + face.h),
                color, 2
            )
            
            label = labels[i] if labels and i < len(labels) else f"{face.confidence:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(
                result,
                (face.x, face.y - th - 10),
                (face.x + tw + 4, face.y),
                color, -1
            )
            
            cv2.putText(
                result, label,
                (face.x + 2, face.y - 5),
                font, font_scale, (0, 0, 0), thickness
            )
        
        # نمایش حالت و تعداد
        mode_text = f"Mode: {self._mode.upper()} | Faces: {len(faces)}"
        cv2.putText(
            result, mode_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        return result
    
    @property
    def mode(self) -> str:
        return self._mode