from abc import ABC, abstractmethod
from typing import List, Optional
from sqlalchemy.orm import Session


class IRepository(ABC):
    """رابط پایه برای همه Repository ها"""
    
    @abstractmethod
    def get_by_id(self, db: Session, id: int):
        pass
    
    @abstractmethod
    def get_all(self, db: Session, skip: int = 0, limit: int = 100):
        pass
    
    @abstractmethod
    def create(self, db: Session, obj):
        pass
    
    @abstractmethod
    def update(self, db: Session, db_obj, obj_in):
        pass
    
    @abstractmethod
    def delete(self, db: Session, id: int):
        pass


class IEmployeeRepository(IRepository):
    """رابط Repository کارکنان"""
    
    @abstractmethod
    def get_by_personnel_code(self, db: Session, code: str):
        pass
    
    @abstractmethod
    def get_active_employees(self, db: Session):
        pass


class ICameraRepository(IRepository):
    """رابط Repository دوربین‌ها"""
    
    @abstractmethod
    def get_active_cameras(self, db: Session):
        pass
    
    @abstractmethod
    def update_status(self, db: Session, camera_id: int, status: str):
        pass


class IAttendanceSessionRepository(IRepository):
    """رابط Repository جلسات حضور"""
    
    @abstractmethod
    def get_active_sessions(self, db: Session):
        pass
    
    @abstractmethod
    def close_session(self, db: Session, session_id: int):
        pass


class IAlertRepository(IRepository):
    """رابط Repository هشدارها"""
    
    @abstractmethod
    def get_unread_alerts(self, db: Session):
        pass
    
    @abstractmethod
    def mark_as_read(self, db: Session, alert_id: int):
        pass