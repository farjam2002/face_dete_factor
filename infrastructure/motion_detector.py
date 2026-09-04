import cv2
import numpy as np
import logging
from typing import Optional, Tuple


logger = logging.getLogger("attendance")


class MotionDetector:
    """
    تشخیص حرکت با روش تفاضل فریم (Frame Differencing).
    سبک و سریع، بدون نیاز به مدل جداگانه.
    """
    
    def __init__(
        self,
        min_motion_area: int = 500,
        threshold: int = 25,
        blur_kernel: int = 21,
        dilate_iterations: int = 2
    ):
        self.min_motion_area = min_motion_area
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.dilate_iterations = dilate_iterations
        
        self._prev_gray: Optional[np.ndarray] = None
        self._last_motion_score: float = 0.0
        self._last_has_motion: bool = False
        self._last_motion_bbox: Optional[Tuple[int, int, int, int]] = None
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """
        تشخیص حرکت در فریم.
        
        Returns:
            (has_motion, motion_score, motion_bbox)
        """
        if frame is None:
            return False, 0.0, None
        
        try:
            # تبدیل به grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # نویزگیری با Gaussian Blur
            gray = cv2.GaussianBlur(
                gray,
                (self.blur_kernel, self.blur_kernel),
                0
            )
            
            # اولین فریم - فقط ذخیره
            if self._prev_gray is None:
                self._prev_gray = gray
                self._last_has_motion = False
                self._last_motion_score = 0.0
                return False, 0.0, None
            
            # تفاضل مطلق با فریم قبلی
            diff = cv2.absdiff(self._prev_gray, gray)
            
            # Threshold
            _, thresh = cv2.threshold(
                diff, self.threshold, 255, cv2.THRESH_BINARY
            )
            
            # Dilate برای پر کردن حفره‌ها
            thresh = cv2.dilate(
                thresh, None, iterations=self.dilate_iterations
            )
            
            # پیدا کردن contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # محاسبه مساحت کل حرکت
            total_motion_area = 0
            largest_contour = None
            largest_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                total_motion_area += area
                
                if area > largest_area:
                    largest_area = area
                    largest_contour = contour
            
            # محاسبه امتیاز حرکت (نسبت مساحت حرکتی به کل فریم)
            frame_area = frame.shape[0] * frame.shape[1]
            motion_score = total_motion_area / frame_area if frame_area > 0 else 0.0
            
            # تصمیم‌گیری
            has_motion = total_motion_area >= self.min_motion_area
            
            # Bounding box بزرگترین ناحیه حرکتی
            motion_bbox = None
            if has_motion and largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                motion_bbox = (x, y, w, h)
            
            # به‌روزرسانی فریم قبلی
            self._prev_gray = gray
            
            # ذخیره نتایج
            self._last_has_motion = has_motion
            self._last_motion_score = motion_score
            self._last_motion_bbox = motion_bbox
            
            return has_motion, motion_score, motion_bbox
        
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False, 0.0, None
    
    def reset(self):
        """بازنشانی تشخیص‌دهنده"""
        self._prev_gray = None
        self._last_has_motion = False
        self._last_motion_score = 0.0
        self._last_motion_bbox = None
    
    def get_last_result(self) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """دریافت آخرین نتیجه"""
        return self._last_has_motion, self._last_motion_score, self._last_motion_bbox