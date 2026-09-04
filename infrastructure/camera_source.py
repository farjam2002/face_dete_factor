import cv2
import logging
import threading
import time
from typing import Optional, Tuple, List
from pathlib import Path

from infrastructure.motion_detector import MotionDetector
from infrastructure.mask_utils import bbox_inside_mask


logger = logging.getLogger("attendance")


class CameraSource:
    """
    منبع تصویر دوربین با پشتیبانی از تشخیص حرکت و ماسک ناحیه فعال.
    """
    
    def __init__(
        self,
        camera_id: int,
        name: str,
        source: str,
        downscale_width: int = 640,
        motion_enabled: bool = True,
        min_motion_area: int = 500,
        mask_points: Optional[List[Tuple[float, float]]] = None
    ):
        self.camera_id = camera_id
        self.name = name
        self.source = source.strip()
        self.downscale_width = downscale_width
        self.motion_enabled = motion_enabled
        self.mask_points = mask_points or []
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[object] = None
        self._last_frame_lock = threading.Lock()
        self._last_frame_jpg: Optional[bytes] = None
        self._is_open = False
        self._last_error: Optional[str] = None
        self._stop_event = threading.Event()
        
        # تشخیص حرکت
        self._motion_detector = MotionDetector(min_motion_area=min_motion_area)
        
        # آمار
        self._last_motion_detected = False
        self._last_motion_score = 0.0
    
    def _resolve_source(self) -> Tuple:
        """تبدیل آدرس ورودی به چیزی که VideoCapture می‌فهمد."""
        source = self.source.lower()
        
        if source.startswith("webcam:"):
            try:
                index = int(source.split(":", 1)[1])
                return index, cv2.CAP_DSHOW
            except ValueError:
                return 0, cv2.CAP_DSHOW
        
        if source.isdigit():
            return int(source), cv2.CAP_DSHOW
        
        if source.startswith(("rtsp://", "http://", "https://")):
            return self.source, cv2.CAP_FFMPEG
        
        return self.source, 0
    
    def connect(self) -> bool:
        """اتصال به دوربین"""
        with self._lock:
            if self._is_open and self._cap is not None:
                return True
            
            try:
                source, backend = self._resolve_source()
                logger.info(
                    f"Camera '{self.name}' (id={self.camera_id}): "
                    f"connecting to {source}"
                )
                
                if backend:
                    self._cap = cv2.VideoCapture(source, backend)
                else:
                    self._cap = cv2.VideoCapture(source)
                
                if not self._cap.isOpened():
                    self._last_error = "Could not open camera"
                    logger.error(
                        f"Camera '{self.name}': could not open source {source}"
                    )
                    return False
                
                if isinstance(source, int):
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self._is_open = True
                self._last_error = None
                self._motion_detector.reset()
                logger.info(f"Camera '{self.name}': connected successfully")
                return True
            
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Camera '{self.name}': connection error: {e}")
                return False
    
    def disconnect(self):
        """قطع اتصال دوربین"""
        with self._lock:
            self._stop_event.set()
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._is_open = False
            self._motion_detector.reset()
            logger.info(f"Camera '{self.name}': disconnected")
    
    def is_open(self) -> bool:
        return self._is_open and self._cap is not None and self._cap.isOpened()
    
    def update_mask(self, mask_points: List[Tuple[float, float]]):
        """به‌روزرسانی نقاط ماسک"""
        self.mask_points = mask_points or []
        logger.info(
            f"Camera '{self.name}': mask updated with {len(self.mask_points)} points"
        )
    
    def read_frame(self) -> Optional[object]:
        """خواندن یک فریم و تشخیص حرکت"""
        with self._lock:
            if not self.is_open():
                return None
            
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    self._is_open = False
                    self._last_error = "Failed to read frame"
                    return None
                
                # کاهش اندازه
                if self.downscale_width > 0:
                    h, w = frame.shape[:2]
                    if w > self.downscale_width:
                        scale = self.downscale_width / w
                        new_w = self.downscale_width
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                
                # تشخیص حرکت
                if self.motion_enabled:
                    has_motion, motion_score, motion_bbox = self._motion_detector.detect(frame)
                    self._last_motion_detected = has_motion
                    self._last_motion_score = motion_score
                else:
                    self._last_motion_detected = True  # اگر غیرفعال، همیشه فرض می‌کنیم حرکت هست
                    self._last_motion_score = 1.0
                
                # ذخیره آخرین فریم
                with self._last_frame_lock:
                    self._last_frame = frame.copy()
                    ok, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if ok:
                        self._last_frame_jpg = buf.tobytes()
                
                return frame
            
            except Exception as e:
                self._last_error = str(e)
                self._is_open = False
                logger.error(f"Camera '{self.name}': read error: {e}")
                return None
    
    def has_motion(self) -> bool:
        """آیا در فریم آخر حرکت تشخیص داده شده؟"""
        return self._last_motion_detected
    
    def get_motion_score(self) -> float:
        """امتیاز حرکت آخرین فریم (0 تا 1)"""
        return self._last_motion_score
    
    def get_last_frame_jpg(self) -> Optional[bytes]:
        """دریافت آخرین فریم به صورت JPEG"""
        with self._last_frame_lock:
            return self._last_frame_jpg
    
    def get_last_frame(self) -> Optional[object]:
        """دریافت آخرین فریم خام"""
        with self._last_frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None
    
    def get_last_error(self) -> Optional[str]:
        return self._last_error
    
    def save_last_frame(self, path: Path) -> bool:
        """ذخیره آخرین فریم در فایل"""
        with self._last_frame_lock:
            if self._last_frame is None:
                return False
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), self._last_frame)
                return True
            except Exception as e:
                logger.error(f"Camera '{self.name}': could not save frame: {e}")
                return False