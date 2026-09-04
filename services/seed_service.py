from infrastructure.database import SessionLocal
from infrastructure.repositories import AreaRepository
from domain.models import Area
import logging

logger = logging.getLogger(__name__)


class SeedService:
    """سرویس برای ایجاد داده‌های اولیه"""
    
    DEFAULT_AREAS = [
        {"name": "ورودی", "description": "محدوده ورودی اصلی"},
        {"name": "خط تولید", "description": "محدوده خط تولید"},
        {"name": "انبار", "description": "محدوده انبار"},
        {"name": "غذاخوری", "description": "محدوده غذاخوری"},
        {"name": "خروجی", "description": "محدوده خروجی"},
    ]
    
    @staticmethod
    def seed_default_areas():
        """ایجاد محدوده‌های پیش‌فرض در صورت عدم وجود"""
        db = SessionLocal()
        try:
            created_count = 0
            
            for area_data in SeedService.DEFAULT_AREAS:
                # بررسی وجود محدوده
                existing = AreaRepository.get_by_name(db, area_data["name"])
                
                if not existing:
                    # ایجاد محدوده جدید
                    area = Area(
                        name=area_data["name"],
                        description=area_data["description"],
                        is_active=True
                    )
                    AreaRepository.create(db, area)
                    created_count += 1
                    logger.info(f"Created default area: {area_data['name']}")
            
            if created_count > 0:
                logger.info(f"Created {created_count} default areas")
            else:
                logger.info("All default areas already exist")
                
        except Exception as e:
            logger.error(f"Error seeding default areas: {e}")
            db.rollback()
        finally:
            db.close()