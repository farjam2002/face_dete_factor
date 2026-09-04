import cv2
import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np


logger = logging.getLogger("attendance")


class FaceGallery:
    """
    مدیریت ذخیره‌سازی و بارگذاری تصاویر چهره کارکنان.
    """
    
    def __init__(self, base_images_dir: Path):
        self.base_dir = base_images_dir / "employees"
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_employee_dir(self, employee_id: int) -> Path:
        """دریافت پوشه تصاویر یک کارمند"""
        employee_dir = self.base_dir / str(employee_id)
        employee_dir.mkdir(parents=True, exist_ok=True)
        return employee_dir
    
    def save_face(
        self,
        employee_id: int,
        image_data: bytes,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        ذخیره تصویر چهره برای یک کارمند.
        چهره به اندازه استاندارد تبدیل می‌شود.
        """
        try:
            # تبدیل bytes به تصویر
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error(f"Could not decode face image for employee {employee_id}")
                return None
            
            # اعتبارسنجی حداقل اندازه
            h, w = img.shape[:2]
            if w < 60 or h < 60:
                logger.warning(
                    f"Face image too small ({w}x{h}) for employee {employee_id}"
                )
            
            # تغییر اندازه به استاندارد
            img = cv2.resize(img, (256, 256))
            
            # تعیین نام فایل
            if not filename:
                import time
                filename = f"face_{int(time.time() * 1000)}.jpg"
            elif not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += ".jpg"
            
            employee_dir = self.get_employee_dir(employee_id)
            file_path = employee_dir / filename
            
            # ذخیره
            cv2.imwrite(str(file_path), img)
            
            logger.info(
                f"Saved face for employee {employee_id}: {file_path}"
            )
            
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving face: {e}")
            return None
    
    def load_faces(self, employee_id: int) -> List[np.ndarray]:
        """بارگذاری همه تصاویر چهره یک کارمند"""
        faces = []
        employee_dir = self.base_dir / str(employee_id)
        
        if not employee_dir.exists():
            return faces
        
        for img_path in employee_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    faces.append(img)
            except Exception as e:
                logger.error(f"Error loading face {img_path}: {e}")
        
        return faces
    
    def load_all_faces(self) -> Dict[int, List[np.ndarray]]:
        """بارگذاری همه چهره‌های همه کارکنان"""
        gallery = {}
        
        if not self.base_dir.exists():
            return gallery
        
        for employee_dir in self.base_dir.iterdir():
            if employee_dir.is_dir():
                try:
                    employee_id = int(employee_dir.name)
                    faces = self.load_faces(employee_id)
                    if faces:
                        gallery[employee_id] = faces
                except ValueError:
                    continue
        
        logger.info(
            f"FaceGallery: loaded {len(gallery)} employees with faces"
        )
        return gallery
    
    def delete_face(self, file_path: str) -> bool:
        """حذف یک تصویر چهره"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted face: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            return False
    
    def delete_all_faces(self, employee_id: int) -> int:
        """حذف همه تصاویر چهره یک کارمند"""
        employee_dir = self.base_dir / str(employee_id)
        count = 0
        
        if employee_dir.exists():
            for img_path in employee_dir.glob("*.jpg"):
                try:
                    img_path.unlink()
                    count += 1
                except Exception:
                    pass
        
        logger.info(
            f"Deleted {count} faces for employee {employee_id}"
        )
        return count
    
    def count_faces(self, employee_id: int) -> int:
        """شمارش تصاویر چهره یک کارمند"""
        employee_dir = self.base_dir / str(employee_id)
        if not employee_dir.exists():
            return 0
        return len(list(employee_dir.glob("*.jpg")))