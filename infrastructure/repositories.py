from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
from domain.models import (
    Employee, Camera, AttendanceSession, Alert, Area,
    UnknownPerson, Sighting, CameraStats, FaceImage
)
from domain.enums import SessionStatus, ReviewStatus


class EmployeeRepository:
    """Repository برای کارکنان"""
    
    @staticmethod
    def get_by_id(db: Session, employee_id: int) -> Optional[Employee]:
        return db.query(Employee).filter(Employee.id == employee_id).first()
    
    @staticmethod
    def get_by_personnel_code(db: Session, code: str) -> Optional[Employee]:
        return db.query(Employee).filter(Employee.personnel_code == code).first()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Employee]:
        return db.query(Employee).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_active_employees(db: Session) -> List[Employee]:
        return db.query(Employee).filter(Employee.is_active == True).all()
    
    @staticmethod
    def create(db: Session, employee: Employee) -> Employee:
        db.add(employee)
        db.commit()
        db.refresh(employee)
        return employee
    
    @staticmethod
    def update(db: Session, employee: Employee) -> Employee:
        db.commit()
        db.refresh(employee)
        return employee
    
    @staticmethod
    def delete(db: Session, employee_id: int) -> bool:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if employee:
            db.delete(employee)
            db.commit()
            return True
        return False


class CameraRepository:
    """Repository برای دوربین‌ها"""
    
    @staticmethod
    def get_by_id(db: Session, camera_id: int) -> Optional[Camera]:
        return db.query(Camera).filter(Camera.id == camera_id).first()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Camera]:
        return db.query(Camera).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_active_cameras(db: Session) -> List[Camera]:
        return db.query(Camera).filter(Camera.is_active == True).all()
    
    @staticmethod
    def create(db: Session, camera: Camera) -> Camera:
        db.add(camera)
        db.commit()
        db.refresh(camera)
        return camera
    
    @staticmethod
    def update(db: Session, camera: Camera) -> Camera:
        db.commit()
        db.refresh(camera)
        return camera
    
    @staticmethod
    def update_status(db: Session, camera_id: int, status: str, error: Optional[str] = None) -> Optional[Camera]:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if camera:
            camera.status = status
            if error:
                camera.last_error = error
            db.commit()
            db.refresh(camera)
        return camera
    
    @staticmethod
    def delete(db: Session, camera_id: int) -> bool:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if camera:
            db.delete(camera)
            db.commit()
            return True
        return False


class AreaRepository:
    """Repository برای محدوده‌ها"""
    
    @staticmethod
    def get_by_id(db: Session, area_id: int) -> Optional[Area]:
        return db.query(Area).filter(Area.id == area_id).first()
    
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[Area]:
        return db.query(Area).filter(Area.name == name).first()
    
    @staticmethod
    def get_all(db: Session) -> List[Area]:
        return db.query(Area).all()
    
    @staticmethod
    def create(db: Session, area: Area) -> Area:
        db.add(area)
        db.commit()
        db.refresh(area)
        return area


class AttendanceSessionRepository:
    """Repository برای جلسات حضور"""
    
    @staticmethod
    def get_by_id(db: Session, session_id: int) -> Optional[AttendanceSession]:
        return db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    
    @staticmethod
    def get_active_sessions(db: Session) -> List[AttendanceSession]:
        return db.query(AttendanceSession).filter(
            AttendanceSession.status == SessionStatus.ACTIVE.value
        ).all()
    
    @staticmethod
    def create(db: Session, session: AttendanceSession) -> AttendanceSession:
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def update(db: Session, session: AttendanceSession) -> AttendanceSession:
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def close_session(db: Session, session_id: int, ended_at) -> Optional[AttendanceSession]:
        session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
        if session:
            session.status = SessionStatus.CLOSED.value
            session.ended_at = ended_at
            if session.first_seen_at:
                duration = (ended_at - session.first_seen_at).total_seconds()
                session.duration_seconds = int(duration)
            db.commit()
            db.refresh(session)
        return session


class AlertRepository:
    """Repository برای هشدارها"""
    
    @staticmethod
    def get_unread_alerts(db: Session, limit: int = 50) -> List[Alert]:
        return db.query(Alert).filter(
            Alert.is_read == False
        ).order_by(Alert.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def create(db: Session, alert: Alert) -> Alert:
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert
    
    @staticmethod
    def mark_as_read(db: Session, alert_id: int) -> bool:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_read = True
            db.commit()
            return True
        return False


class UnknownPersonRepository:
    """Repository برای افراد ناشناس"""
    
    @staticmethod
    def get_by_id(db: Session, unknown_id: int) -> Optional[UnknownPerson]:
        return db.query(UnknownPerson).filter(UnknownPerson.id == unknown_id).first()
    
    @staticmethod
    def get_by_code(db: Session, code: str) -> Optional[UnknownPerson]:
        return db.query(UnknownPerson).filter(UnknownPerson.code == code).first()
    
    @staticmethod
    def create(db: Session, unknown: UnknownPerson) -> UnknownPerson:
        db.add(unknown)
        db.commit()
        db.refresh(unknown)
        return unknown
    
    @staticmethod
    def get_next_code(db: Session) -> str:
        """تولید کد بعدی برای ناشناس"""
        # پیدا کردن آخرین کد
        last_unknown = db.query(UnknownPerson).order_by(UnknownPerson.id.desc()).first()
        if not last_unknown:
            return "UNKNOWN-000001"
        
        # استخراج شماره از کد قبلی
        try:
            last_num = int(last_unknown.code.split("-")[1])
            next_num = last_num + 1
            return f"UNKNOWN-{next_num:06d}"
        except:
            return f"UNKNOWN-{(last_unknown.id + 1):06d}"


class SightingRepository:
    """Repository برای مشاهدات"""
    
    @staticmethod
    def create(db: Session, sighting: Sighting) -> Sighting:
        db.add(sighting)
        db.commit()
        db.refresh(sighting)
        return sighting
    
    @staticmethod
    def get_by_session(db: Session, session_id: int) -> List[Sighting]:
        return db.query(Sighting).filter(
            Sighting.session_id == session_id
        ).order_by(Sighting.seen_at.desc()).all()


class CameraStatsRepository:
    """Repository برای آمار دوربین‌ها"""
    
    @staticmethod
    def get_by_camera(db: Session, camera_id: int) -> Optional[CameraStats]:
        return db.query(CameraStats).filter(CameraStats.camera_id == camera_id).first()
    
    @staticmethod
    def create(db: Session, stats: CameraStats) -> CameraStats:
        db.add(stats)
        db.commit()
        db.refresh(stats)
        return stats
    
    @staticmethod
    def update(db: Session, stats: CameraStats) -> CameraStats:
        db.commit()
        db.refresh(stats)
        return stats


class FaceImageRepository:
    """Repository برای تصاویر چهره"""
    
    @staticmethod
    def get_by_employee(db: Session, employee_id: int) -> List[FaceImage]:
        return db.query(FaceImage).filter(FaceImage.employee_id == employee_id).all()
    
    @staticmethod
    def create(db: Session, face_image: FaceImage) -> FaceImage:
        db.add(face_image)
        db.commit()
        db.refresh(face_image)
        return face_image
    
    @staticmethod
    def delete(db: Session, face_id: int) -> bool:
        face = db.query(FaceImage).filter(FaceImage.id == face_id).first()
        if face:
            db.delete(face)
            db.commit()
            return True
        return False