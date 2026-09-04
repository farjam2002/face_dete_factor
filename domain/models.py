from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship, backref
from infrastructure.database import Base
from domain.enums import (
    PersonType, SessionStatus, ReviewStatus, ShiftType,
    CameraStatus, AlertType, AlertLevel, ActionType
)


class Employee(Base):
    """جدول کارکنان"""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    personnel_code = Column(String(50), unique=True, nullable=False, index=True, comment="کد پرسنلی")
    full_name = Column(String(200), nullable=False, comment="نام و نام خانوادگی")
    is_active = Column(Boolean, default=True, nullable=False, index=True, comment="فعال/غیرفعال")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # روابط
    face_images = relationship("FaceImage", back_populates="employee", cascade="all, delete-orphan")
    sessions = relationship("AttendanceSession", back_populates="employee")
    sightings = relationship("Sighting", back_populates="employee")

    __table_args__ = (
        Index("idx_employee_active", "is_active"),
        Index("idx_employee_code", "personnel_code"),
    )


class FaceImage(Base):
    """جدول تصاویر چهره کارکنان"""
    __tablename__ = "face_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String(500), nullable=False, comment="مسیر فایل تصویر")
    embedding = Column(Text, nullable=True, comment="ویژگی‌های چهره به صورت JSON")
    quality_score = Column(Float, default=0.0, comment="امتیاز کیفیت چهره")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # روابط
    employee = relationship("Employee", back_populates="face_images")

    __table_args__ = (
        Index("idx_face_employee", "employee_id"),
    )


class Area(Base):
    """جدول محدوده‌ها (ورودی، خط تولید، انبار و ...)"""
    __tablename__ = "areas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, comment="نام محدوده")
    description = Column(String(500), nullable=True, comment="توضیحات")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # روابط
    cameras = relationship("Camera", back_populates="area")
    sessions = relationship("AttendanceSession", back_populates="area")
    sightings = relationship("Sighting", back_populates="area")


class Camera(Base):
    """جدول دوربین‌ها"""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="نام دوربین")
    rtsp_url = Column(String(500), nullable=False, comment="آدرس RTSP")
    area_id = Column(Integer, ForeignKey("areas.id"), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    status = Column(String(20), default=CameraStatus.OFFLINE.value, nullable=False)
    mask_points = Column(JSON, nullable=True, comment="نقاط ماسک به صورت JSON")
    settings = Column(JSON, nullable=True, comment="تنظیمات اضافی")
    last_success_at = Column(DateTime, nullable=True, comment="آخرین پردازش موفق")
    last_error = Column(Text, nullable=True, comment="آخرین خطا")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # روابط
    area = relationship("Area", back_populates="cameras")
    stats = relationship("CameraStats", back_populates="camera", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("AttendanceSession", back_populates="camera")
    sightings = relationship("Sighting", back_populates="camera")
    alerts = relationship("Alert", back_populates="camera")

    __table_args__ = (
        Index("idx_camera_status", "status"),
        Index("idx_camera_active", "is_active"),
    )


class CameraStats(Base):
    """جدول آمار هر دوربین"""
    __tablename__ = "camera_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), unique=True, nullable=False)
    frames_read = Column(Integer, default=0, comment="تعداد فریم خوانده‌شده")
    frames_processed = Column(Integer, default=0, comment="تعداد فریم پردازش‌شده")
    faces_detected = Column(Integer, default=0, comment="تعداد چهره شناسایی‌شده")
    identities_matched = Column(Integer, default=0, comment="تعداد هویت تطبیق‌شده")
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # روابط
    camera = relationship("Camera", back_populates="stats")


class UnknownPerson(Base):
    """جدول افراد ناشناس"""
    __tablename__ = "unknown_persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True, comment="کد ناشناس مثل UNKNOWN-000001")
    primary_image_path = Column(String(500), nullable=True, comment="مسیر تصویر اصلی")
    notes = Column(Text, nullable=True, comment="یادداشت")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    merged_into_id = Column(Integer, ForeignKey("unknown_persons.id"), nullable=True, comment="اگر ادغام شده باشد")

    # رابطه خود-ارجاعی برای ادغام ناشناس‌ها
    merged_into = relationship(
        "UnknownPerson",
        remote_side=id,
        backref=backref("merged_unknowns", remote_side=merged_into_id)
    )

    # روابط با جلسات و مشاهدات (foreign_keys را در مدل‌های مقصد تعریف می‌کنیم)
    sessions = relationship(
        "AttendanceSession",
        back_populates="unknown",
        foreign_keys="[AttendanceSession.unknown_id]"
    )
    sightings = relationship(
        "Sighting",
        back_populates="unknown",
        foreign_keys="[Sighting.unknown_id]"
    )

    __table_args__ = (
        Index("idx_unknown_code", "code"),
    )


class AttendanceSession(Base):
    """جدول جلسات حضور"""
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_type = Column(String(20), nullable=False, index=True, comment="نوع فرد")
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True, index=True)
    unknown_id = Column(Integer, ForeignKey("unknown_persons.id"), nullable=True, index=True)
    temp_track_id = Column(String(100), nullable=True, comment="شناسه موقت ردیابی")
    area_id = Column(Integer, ForeignKey("areas.id"), nullable=False, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    first_seen_at = Column(DateTime, nullable=False, index=True, comment="اولین زمان مشاهده")
    last_seen_at = Column(DateTime, nullable=False, index=True, comment="آخرین زمان مشاهده")
    ended_at = Column(DateTime, nullable=True, comment="زمان پایان جلسه")
    duration_seconds = Column(Integer, default=0, comment="مدت حضور به ثانیه")
    status = Column(String(20), default=SessionStatus.ACTIVE.value, nullable=False, index=True)
    confidence = Column(Float, default=0.0, comment="میانگین اطمینان")
    review_status = Column(String(20), default=ReviewStatus.AUTO.value, nullable=False, index=True)
    shift = Column(String(20), default=ShiftType.NONE.value, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # روابط
    employee = relationship("Employee", back_populates="sessions")
    unknown = relationship("UnknownPerson", back_populates="sessions", foreign_keys=[unknown_id])
    area = relationship("Area", back_populates="sessions")
    camera = relationship("Camera", back_populates="sessions")
    sightings = relationship("Sighting", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_session_employee", "employee_id"),
        Index("idx_session_unknown", "unknown_id"),
        Index("idx_session_area", "area_id"),
        Index("idx_session_status", "status"),
        Index("idx_session_first_seen", "first_seen_at"),
        Index("idx_session_last_seen", "last_seen_at"),
        Index("idx_session_review", "review_status"),
    )


class Sighting(Base):
    """جدول مشاهدات (هر بار دیده شدن یک فرد)"""
    __tablename__ = "sightings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    area_id = Column(Integer, ForeignKey("areas.id"), nullable=False, index=True)
    person_type = Column(String(20), nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True, index=True)
    unknown_id = Column(Integer, ForeignKey("unknown_persons.id"), nullable=True, index=True)
    temp_track_id = Column(String(100), nullable=True)
    seen_at = Column(DateTime, nullable=False, index=True, comment="زمان مشاهده")
    bbox = Column(String(100), nullable=True, comment="مختصات جعبه چهره")
    confidence = Column(Float, default=0.0, comment="اطمینان تشخیص")
    quality_score = Column(Float, default=0.0, comment="کیفیت چهره")
    frame_path = Column(String(500), nullable=True, comment="مسیر فریم ذخیره‌شده")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # روابط
    session = relationship("AttendanceSession", back_populates="sightings")
    camera = relationship("Camera", back_populates="sightings")
    area = relationship("Area", back_populates="sightings")
    employee = relationship("Employee", back_populates="sightings")
    unknown = relationship("UnknownPerson", back_populates="sightings", foreign_keys=[unknown_id])

    __table_args__ = (
        Index("idx_sighting_session", "session_id"),
        Index("idx_sighting_seen_at", "seen_at"),
        Index("idx_sighting_employee", "employee_id"),
        Index("idx_sighting_unknown", "unknown_id"),
    )


class Alert(Base):
    """جدول هشدارها"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(50), nullable=False, index=True, comment="نوع هشدار")
    level = Column(String(20), nullable=False, index=True, comment="سطح هشدار")
    message = Column(Text, nullable=False, comment="پیام هشدار")
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    unknown_id = Column(Integer, ForeignKey("unknown_persons.id"), nullable=True, index=True)
    is_read = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # روابط
    camera = relationship("Camera", back_populates="alerts")
    unknown = relationship("UnknownPerson")

    __table_args__ = (
        Index("idx_alert_type", "type"),
        Index("idx_alert_level", "level"),
        Index("idx_alert_created", "created_at"),
        Index("idx_alert_read", "is_read"),
    )


class AuditLog(Base):
    """جدول لاگ‌های عملیاتی"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(50), nullable=False, index=True, comment="نوع عملیات")
    entity_type = Column(String(50), nullable=True, comment="نوع موجودیت")
    entity_id = Column(String(50), nullable=True, comment="شناسه موجودیت")
    details = Column(Text, nullable=True, comment="جزئیات")
    user_id = Column(String(50), nullable=True, comment="کاربر انجام‌دهنده")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_created", "created_at"),
    )


class ReviewCase(Base):
    """جدول موارد بررسی دستی"""
    __tablename__ = "review_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False, index=True)
    sighting_id = Column(Integer, ForeignKey("sightings.id"), nullable=True, index=True)
    status = Column(String(20), default=ReviewStatus.PENDING.value, nullable=False, index=True)
    reviewer_notes = Column(Text, nullable=True, comment="یادداشت بررسی‌کننده")
    corrected_employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # روابط
    session = relationship("AttendanceSession")
    sighting = relationship("Sighting")
    corrected_employee = relationship("Employee")

    __table_args__ = (
        Index("idx_review_status", "status"),
    )