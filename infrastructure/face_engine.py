import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path


logger = logging.getLogger("attendance")


class FaceEngine:
    """
    موتور شناسایی چهره.
    دو حالت:
    1. LBPH (دقیق‌تر) - نیاز به opencv-contrib
    2. SimpleDistance (ساده‌تر) - همیشه در دسترس
    """
    
    # برچسب‌های رزرو شده
    LABEL_UNKNOWN = -1
    LABEL_UNCERTAIN = -2
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.72,
        low_confidence_threshold: float = 0.45
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        
        self._recognizer = None
        self._mode = "none"
        self._gallery: Dict[int, List[np.ndarray]] = {}
        self._needs_rebuild = False
        
        # تلاش برای ساخت شناساگر
        self._init_recognizer()
    
    def _init_recognizer(self):
        """راه‌اندازی شناساگر"""
        try:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create()
            self._mode = "lbph"
            logger.info("FaceEngine: LBPH mode initialized")
        except AttributeError:
            logger.warning(
                "FaceEngine: cv2.face not available, "
                "falling back to simple distance mode"
            )
            self._mode = "simple"
    
    @property
    def mode(self) -> str:
        return self._mode
    
    def mark_gallery_dirty(self):
        """علامت‌گذاری گالری برای بازسازی"""
        self._needs_rebuild = True
        logger.info("FaceEngine: gallery marked for rebuild")
    
    def needs_rebuild(self) -> bool:
        return self._needs_rebuild
    
    def set_gallery(self, gallery: Dict[int, List[np.ndarray]]):
        """
        تنظیم گالری چهره‌ها.
        gallery: {employee_id: [face_image, ...]}
        """
        self._gallery = gallery
        self._needs_rebuild = True
    
    def rebuild(self):
        """بازسازی شناساگر با گالری فعلی"""
        if self._mode == "none":
            return
        
        images = []
        labels = []
        
        for employee_id, face_list in self._gallery.items():
            for face_img in face_list:
                if face_img is not None and face_img.size > 0:
                    # تبدیل به grayscale و اندازه استاندارد
                    face_gray = self._prepare_face(face_img)
                    images.append(face_gray)
                    labels.append(int(employee_id))
        
        if not images:
            logger.info("FaceEngine: gallery is empty, nothing to train")
            if self._mode == "lbph" and self._recognizer is not None:
                # ریست کردن شناساگر
                self._recognizer = cv2.face.LBPHFaceRecognizer_create()
            self._needs_rebuild = False
            return
        
        try:
            if self._mode == "lbph":
                labels_array = np.array(labels, dtype=np.int32)
                self._recognizer.train(images, labels_array)
                logger.info(
                    f"FaceEngine: LBPH trained with {len(images)} images, "
                    f"{len(self._gallery)} employees"
                )
            
            self._needs_rebuild = False
        
        except Exception as e:
            logger.error(f"FaceEngine: rebuild error: {e}")
    
    def _prepare_face(self, face: np.ndarray) -> np.ndarray:
        """آماده‌سازی چهره برای پردازش"""
        if face.ndim == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        face = cv2.resize(face, (128, 128))
        face = cv2.equalizeHist(face)
        
        return face
    
    def recognize(self, face: np.ndarray) -> Tuple[int, float]:
        """
        شناسایی چهره.
        
        Returns:
            (label, confidence)
            label: employee_id یا LABEL_UNKNOWN یا LABEL_UNCERTAIN
            confidence: 0 تا 1 (بالاتر = مطمئن‌تر)
        """
        if self._mode == "none":
            return self.LABEL_UNKNOWN, 0.0
        
        # بازسازی در صورت نیاز
        if self._needs_rebuild:
            self.rebuild()
        
        if self._mode == "lbph":
            return self._recognize_lbph(face)
        else:
            return self._recognize_simple(face)
    
    def _recognize_lbph(self, face: np.ndarray) -> Tuple[int, float]:
        """شناسایی با LBPH"""
        if self._recognizer is None or not self._gallery:
            return self.LABEL_UNKNOWN, 0.0
        
        try:
            face_gray = self._prepare_face(face)
            label, confidence = self._recognizer.predict(face_gray)
            
            # تبدیل فاصله به اطمینان (در LBPH فاصله کمتر = بهتر)
            # معمولاً 0 تا 100
            confidence_01 = max(0.0, 1.0 - (confidence / 100.0))
            
            if confidence_01 >= self.high_confidence_threshold:
                return int(label), confidence_01
            elif confidence_01 >= self.low_confidence_threshold:
                return self.LABEL_UNCERTAIN, confidence_01
            else:
                return self.LABEL_UNKNOWN, confidence_01
        
        except Exception as e:
            logger.error(f"LBPH recognition error: {e}")
            return self.LABEL_UNKNOWN, 0.0
    
    def _recognize_simple(self, face: np.ndarray) -> Tuple[int, float]:
        """شناسایی ساده با فاصله اقلیدسی (fallback)"""
        if not self._gallery:
            return self.LABEL_UNKNOWN, 0.0
        
        try:
            face_gray = self._prepare_face(face).astype(np.float32)
            
            best_label = self.LABEL_UNKNOWN
            best_score = 0.0
            
            for employee_id, face_list in self._gallery.items():
                for ref_face in face_list:
                    ref_gray = self._prepare_face(ref_face).astype(np.float32)
                    
                    # فاصله اقلیدسی نرمال‌شده
                    diff = np.linalg.norm(face_gray - ref_gray)
                    max_diff = np.linalg.norm(
                        np.ones_like(face_gray) * 255
                    )
                    similarity = max(0.0, 1.0 - (diff / max_diff))
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_label = employee_id
            
            if best_score >= self.high_confidence_threshold:
                return best_label, best_score
            elif best_score >= self.low_confidence_threshold:
                return self.LABEL_UNCERTAIN, best_score
            else:
                return self.LABEL_UNKNOWN, best_score
        
        except Exception as e:
            logger.error(f"Simple recognition error: {e}")
            return self.LABEL_UNKNOWN, 0.0