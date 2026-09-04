import logging
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session

from infrastructure.database import SessionLocal
from domain.models import Employee, FaceImage
from infrastructure.face_gallery import FaceGallery
from infrastructure.face_engine import FaceEngine


logger = logging.getLogger("attendance")


class EmployeeService:
    """سرویس مدیریت کارکنان و چهره‌ها"""
    
    def __init__(
        self,
        base_dir: Path,
        config: dict,
        face_gallery: FaceGallery,
        face_engine: FaceEngine
    ):
        self.base_dir = base_dir
        self.config = config
        self.face_gallery = face_gallery
        self.face_engine = face_engine
    
    def rebuild_gallery(self):
        """بارگذاری گالری و بازسازی شناساگر"""
        logger.info("EmployeeService: rebuilding face gallery...")
        gallery = self.face_gallery.load_all_faces()
        self.face_engine.set_gallery(gallery)
        self.face_engine.rebuild()
        logger.info("EmployeeService: gallery rebuild complete")
    
    def add_face(
        self,
        employee_id: int,
        image_data: bytes
    ) -> Optional[FaceImage]:
        """افزودن چهره به یک کارمند"""
        db = SessionLocal()
        try:
            # ذخیره فایل
            file_path = self.face_gallery.save_face(employee_id, image_data)
            if not file_path:
                return None
            
            # ذخیره مسیر نسبی در دیتابیس
            relative_path = str(file_path.relative_to(
                self.base_dir / self.config["app"]["data_dir"]
            ))
            
            face_image = FaceImage(
                employee_id=employee_id,
                file_path=relative_path
            )
            
            db.add(face_image)
            db.commit()
            db.refresh(face_image)
            
            # علامت‌گذاری برای بازسازی گالری
            self.face_engine.mark_gallery_dirty()
            
            logger.info(
                f"EmployeeService: added face {face_image.id} "
                f"to employee {employee_id}"
            )
            
            return face_image
        
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def delete_face(self, face_id: int) -> bool:
        """حذف یک چهره"""
        db = SessionLocal()
        try:
            face_image = db.query(FaceImage).filter(
                FaceImage.id == face_id
            ).first()
            
            if not face_image:
                return False
            
            # حذف فایل فیزیکی
            full_path = (
                self.base_dir / self.config["app"]["data_dir"] /
                face_image.file_path
            )
            self.face_gallery.delete_face(str(full_path))
            
            # حذف از دیتابیس
            db.delete(face_image)
            db.commit()
            
            # علامت‌گذاری برای بازسازی گالری
            self.face_engine.mark_gallery_dirty()
            
            logger.info(f"EmployeeService: deleted face {face_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_face_count(self, employee_id: int) -> int:
        """شمارش چهره‌های یک کارمند"""
        return self.face_gallery.count_faces(employee_id)
    
    def delete_employee_faces(self, employee_id: int) -> int:
        """حذف همه چهره‌های یک کارمند"""
        db = SessionLocal()
        try:
            # حذف رکوردهای دیتابیس
            db.query(FaceImage).filter(
                FaceImage.employee_id == employee_id
            ).delete()
            db.commit()
            
            # حذف فایل‌ها
            count = self.face_gallery.delete_all_faces(employee_id)
            
            self.face_engine.mark_gallery_dirty()
            return count
        
        except Exception as e:
            logger.error(f"Error deleting employee faces: {e}")
            db.rollback()
            return 0
        finally:
            db.close()