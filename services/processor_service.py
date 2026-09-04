import logging
from typing import List, Optional, Tuple

from infrastructure.face_detector import FaceDetector
from infrastructure.face_engine import FaceEngine
from services.attendance_service import AttendanceService
from infrastructure.database import SessionLocal
from domain.enums import PersonType
from domain.models import Employee, UnknownPerson


logger = logging.getLogger("attendance")


class ProcessorService:
    """
    سرویس پردازش فریم‌ها.
    
    مسئولیت:
    - تشخیص چهره در فریم
    - شناسایی هویت (کارمند / ناشناس)
    - ثبت در سیستم حضور (اختیاری)
    - برگرداندن نتایج برای نمایش
    """
    
    def __init__(
        self,
        face_detector: FaceDetector,
        face_engine: FaceEngine,
        attendance_service: AttendanceService
    ):
        self.face_detector = face_detector
        self.face_engine = face_engine
        self.attendance_service = attendance_service
    
    def identify_faces(
        self,
        faces: list,
        frame
    ) -> List[Tuple[PersonType, Optional[int], str, float]]:
        """
        شناسایی هویت چهره‌های تشخیص داده شده.
        
        Returns:
            لیست (person_type, identity_id, label_text, confidence)
        """
        results = []
        
        for face in faces:
            if not face.is_usable:
                results.append((
                    PersonType.UNKNOWN,
                    None,
                    "Low Quality",
                    0.0
                ))
                continue
            
            # استخراج چهره
            face_crop = frame[face.y:face.y+face.h, face.x:face.x+face.w]
            if face_crop.size == 0:
                results.append((
                    PersonType.UNKNOWN,
                    None,
                    "?",
                    0.0
                ))
                continue
            
            # شناسایی با FaceEngine
            try:
                label, confidence = self.face_engine.recognize(face_crop)
                
                if label == self.face_engine.LABEL_UNKNOWN:
                    results.append((
                        PersonType.UNKNOWN,
                        None,
                        "Unknown",
                        confidence
                    ))
                
                elif label == self.face_engine.LABEL_UNCERTAIN:
                    results.append((
                        PersonType.UNKNOWN,
                        None,
                        "Uncertain",
                        confidence
                    ))
                
                else:
                    # کارمند شناسایی شده
                    employee = self._get_employee(label)
                    if employee and employee.is_active:
                        results.append((
                            PersonType.EMPLOYEE,
                            label,
                            employee.full_name,
                            confidence
                        ))
                    else:
                        results.append((
                            PersonType.UNKNOWN,
                            None,
                            "Unknown",
                            confidence
                        ))
            
            except Exception as e:
                logger.error(f"Error identifying face: {e}")
                results.append((
                    PersonType.UNKNOWN,
                    None,
                    "Error",
                    0.0
                ))
        
        return results
    
    def _get_employee(self, employee_id: int):
        """گرفتن کارمند از دیتابیس"""
        try:
            db = SessionLocal()
            try:
                return db.query(Employee).filter(Employee.id == employee_id).first()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error getting employee: {e}")
            return None
    
    def process_frame(
        self,
        camera_id: int,
        area_id: Optional[int],
        frame,
        register_attendance: bool = True
    ) -> Tuple[List, List[Tuple[PersonType, Optional[int], str]]]:
        """
        پردازش یک فریم.
        
        Args:
            camera_id: شناسه دوربین
            area_id: شناسه محدوده
            frame: فریم OpenCV
            register_attendance: آیا در سیستم حضور ثبت شود؟
        
        Returns:
            (faces, identity_results)
            - faces: لیست DetectedFace
            - identity_results: لیست (person_type, identity_id, label_text)
        """
        # 1. تشخیص چهره
        faces = self.face_detector.detect(frame)
        
        # 2. شناسایی هویت (برای نمایش)
        identity_results_full = self.identify_faces(faces, frame)
        
        # تبدیل به فرمت ساده (بدون confidence برای سازگاری)
        identity_results = [
            (r[0], r[1], r[2]) for r in identity_results_full
        ]
        
        # 3. ثبت حضور (اختیاری)
        if register_attendance and faces:
            try:
                self.attendance_service.process_detections(
                    camera_id=camera_id,
                    area_id=area_id,
                    faces=faces,
                    frame=frame
                )
            except Exception as e:
                logger.error(f"Error in attendance processing: {e}")
        
        return faces, identity_results
    
    def draw_results(
        self,
        frame,
        faces,
        identity_results: List[Tuple[PersonType, Optional[int], str]]
    ):
        """رسم نتایج روی فریم"""
        labels = []
        
        for i, face in enumerate(faces):
            if i < len(identity_results):
                person_type, identity_id, label_text = identity_results[i]
                
                if not face.is_usable:
                    labels.append("Low Quality")
                elif person_type == PersonType.EMPLOYEE:
                    labels.append(f"✓ {label_text}")
                elif person_type == PersonType.UNKNOWN:
                    labels.append(f"? {label_text}")
                else:
                    labels.append(f"~ {label_text}")
            else:
                labels.append("?")
        
        return self.face_detector.draw_faces(frame, faces, labels)