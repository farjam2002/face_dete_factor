import logging
import threading
import cv2
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from infrastructure.database import SessionLocal
from domain.models import UnknownPerson, Employee, FaceImage


logger = logging.getLogger("attendance")


class UnknownService:
    """
    سرویس مدیریت افراد ناشناس.
    """
    
    def __init__(self, base_dir: Path, config: dict):
        self.base_dir = base_dir
        self.config = config
        
        self._images_dir = (
            base_dir / config["app"]["data_dir"] /
            config["storage"]["images_dir"] / "unknowns"
        )
        self._images_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        
        logger.info(f"UnknownService initialized, images dir: {self._images_dir}")
    
    def get_or_create_unknown(
        self,
        face_crop=None,
        frame=None
    ) -> Optional[int]:
        """
        پیدا کردن یا ایجاد یک ناشناس جدید.
        """
        with self._lock:
            db = SessionLocal()
            try:
                # پیدا کردن آخرین ناشناس
                last_unknown = db.query(UnknownPerson).order_by(
                    UnknownPerson.id.desc()
                ).first()
                
                # تولید کد جدید
                if last_unknown:
                    try:
                        last_num = int(last_unknown.code.split("-")[1])
                        new_num = last_num + 1
                    except (IndexError, ValueError):
                        new_num = last_unknown.id + 1
                else:
                    new_num = 1
                
                new_code = f"UNKNOWN-{new_num:06d}"
                
                # ایجاد ناشناس جدید
                unknown = UnknownPerson(code=new_code)
                
                # ذخیره تصویر اگر موجود باشد
                if face_crop is not None and face_crop.size > 0:
                    image_path = self._save_unknown_image(
                        unknown_id=new_num,
                        face_crop=face_crop
                    )
                    if image_path:
                        unknown.primary_image_path = image_path
                        logger.info(f"Saved image for unknown {new_num}: {image_path}")
                    else:
                        logger.warning(f"Failed to save image for unknown {new_num}")
                
                db.add(unknown)
                db.commit()
                db.refresh(unknown)
                
                logger.info(
                    f"✅ New unknown created: id={unknown.id}, code={unknown.code}"
                )
                
                return unknown.id
            
            except Exception as e:
                logger.error(f"❌ Error creating unknown: {e}")
                import traceback
                logger.error(traceback.format_exc())
                db.rollback()
                return None
            finally:
                db.close()
    
    def _save_unknown_image(
        self,
        unknown_id: int,
        face_crop
    ) -> Optional[str]:
        """ذخیره تصویر چهره ناشناس"""
        try:
            filename = f"unknown_{unknown_id}.jpg"
            file_path = self._images_dir / filename
            
            # تبدیل به RGB برای ذخیره
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_crop
            
            # ذخیره
            success = cv2.imwrite(str(file_path), face_rgb)
            
            if not success:
                logger.error(f"cv2.imwrite returned False for {file_path}")
                return None
            
            # بررسی وجود فایل
            if not file_path.exists():
                logger.error(f"File not created: {file_path}")
                return None
            
            # برگرداندن مسیر نسبی
            relative_path = f"images/unknowns/{filename}"
            logger.info(f"Image saved successfully: {file_path} ({file_path.stat().st_size} bytes)")
            return relative_path
        
        except Exception as e:
            logger.error(f"❌ Error saving unknown image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def assign_to_employee(
        self,
        unknown_id: int,
        employee_id: int,
        employee_service=None
    ) -> bool:
        """اختصاص یک ناشناس به کارمند."""
        db = SessionLocal()
        try:
            unknown = db.query(UnknownPerson).filter(
                UnknownPerson.id == unknown_id
            ).first()
            
            if not unknown:
                logger.error(f"Unknown {unknown_id} not found")
                return False
            
            employee = db.query(Employee).filter(
                Employee.id == employee_id
            ).first()
            
            if not employee:
                logger.error(f"Employee {employee_id} not found")
                return False
            
            # انتقال تصویر چهره ناشناس به کارمند
            if unknown.primary_image_path and employee_service:
                try:
                    full_path = (
                        self.base_dir / self.config["app"]["data_dir"] /
                        unknown.primary_image_path
                    )
                    
                    if full_path.exists():
                        face_img = cv2.imread(str(full_path))
                        
                        if face_img is not None:
                            _, buffer = cv2.imencode('.jpg', face_img)
                            image_data = buffer.tobytes()
                            
                            employee_service.add_face(employee_id, image_data)
                            logger.info(
                                f"Transferred face from unknown {unknown_id} "
                                f"to employee {employee_id}"
                            )
                except Exception as e:
                    logger.error(f"Error transferring face: {e}")
            
            # به‌روزرسانی sessions مرتبط
            from domain.models import AttendanceSession, Sighting
            
            db.query(AttendanceSession).filter(
                AttendanceSession.unknown_id == unknown_id
            ).update({
                "unknown_id": None,
                "employee_id": employee_id,
                "person_type": "employee"
            })
            
            db.query(Sighting).filter(
                Sighting.unknown_id == unknown_id
            ).update({
                "unknown_id": None,
                "employee_id": employee_id,
                "person_type": "employee"
            })
            
            # حذف ناشناس
            db.delete(unknown)
            db.commit()
            
            logger.info(
                f"✅ Unknown {unknown_id} assigned to employee {employee_id}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error assigning unknown to employee: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def merge_unknowns(
        self,
        source_unknown_id: int,
        target_unknown_id: int
    ) -> bool:
        """ادغام دو ناشناس."""
        db = SessionLocal()
        try:
            source = db.query(UnknownPerson).filter(
                UnknownPerson.id == source_unknown_id
            ).first()
            
            target = db.query(UnknownPerson).filter(
                UnknownPerson.id == target_unknown_id
            ).first()
            
            if not source or not target:
                logger.error("Unknown not found for merge")
                return False
            
            if source_unknown_id == target_unknown_id:
                logger.error("Cannot merge unknown with itself")
                return False
            
            from domain.models import AttendanceSession, Sighting
            
            db.query(AttendanceSession).filter(
                AttendanceSession.unknown_id == source_unknown_id
            ).update({"unknown_id": target_unknown_id})
            
            db.query(Sighting).filter(
                Sighting.unknown_id == source_unknown_id
            ).update({"unknown_id": target_unknown_id})
            
            source.merged_into_id = target_unknown_id
            
            db.commit()
            
            logger.info(
                f"✅ Merged unknown {source_unknown_id} into {target_unknown_id}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error merging unknowns: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def delete_unknown(self, unknown_id: int) -> bool:
        """حذف یک ناشناس."""
        db = SessionLocal()
        try:
            unknown = db.query(UnknownPerson).filter(
                UnknownPerson.id == unknown_id
            ).first()
            
            if not unknown:
                return False
            
            from domain.models import AttendanceSession
            
            session_count = db.query(AttendanceSession).filter(
                AttendanceSession.unknown_id == unknown_id
            ).count()
            
            if session_count > 0:
                logger.warning(
                    f"Cannot delete unknown {unknown_id}: "
                    f"has {session_count} sessions"
                )
                return False
            
            if unknown.primary_image_path:
                try:
                    full_path = (
                        self.base_dir / self.config["app"]["data_dir"] /
                        unknown.primary_image_path
                    )
                    if full_path.exists():
                        full_path.unlink()
                except Exception:
                    pass
            
            db.delete(unknown)
            db.commit()
            
            logger.info(f"✅ Unknown {unknown_id} deleted")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting unknown: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_unknown_by_id(self, unknown_id: int) -> Optional[UnknownPerson]:
        """گرفتن ناشناس بر اساس شناسه"""
        db = SessionLocal()
        try:
            return db.query(UnknownPerson).filter(
                UnknownPerson.id == unknown_id
            ).first()
        finally:
            db.close()
    
    def get_all_unknowns(self) -> List[UnknownPerson]:
        """گرفتن همه ناشناس‌ها"""
        db = SessionLocal()
        try:
            return db.query(UnknownPerson).filter(
                UnknownPerson.merged_into_id == None
            ).order_by(UnknownPerson.id.desc()).all()
        finally:
            db.close()