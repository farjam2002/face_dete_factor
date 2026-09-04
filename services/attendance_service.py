import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from infrastructure.database import SessionLocal
from domain.models import (
    AttendanceSession, Sighting, Employee, UnknownPerson
)
from domain.enums import (
    PersonType, SessionStatus, ReviewStatus, ShiftType
)
from infrastructure.face_engine import FaceEngine


logger = logging.getLogger("attendance")


class AttendanceService:
    """سرویس اصلی منطق حضور و غیاب."""
    
    def __init__(
        self,
        config: dict,
        base_dir: Path,
        face_engine: FaceEngine,
        unknown_service=None
    ):
        self.config = config
        self.base_dir = base_dir
        self.face_engine = face_engine
        self.unknown_service = unknown_service
        
        # تنظیمات حضور
        attendance_config = config["attendance"]
        self.session_close_timeout = attendance_config["session_close_timeout_seconds"]
        self.duplicate_event_interval = attendance_config["duplicate_event_interval_seconds"]
        
        # تنظیمات شناسایی
        recognition_config = config["recognition"]
        self.high_conf_threshold = recognition_config["high_confidence_threshold"]
        self.low_conf_threshold = recognition_config["low_confidence_threshold"]
        
        # cache برای جلوگیری از duplicate
        self._last_event_time: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        # mapping از temp_track_id به unknown_id
        self._temp_to_unknown: Dict[str, int] = {}
        
        logger.info(
            f"AttendanceService initialized. "
            f"Unknown service: {'✅ available' if unknown_service else '❌ NOT available'}"
        )
    
    def process_detections(
        self,
        camera_id: int,
        area_id: Optional[int],
        faces: list,
        frame
    ) -> List[Tuple[PersonType, Optional[int], str]]:
        """پردازش چهره‌های تشخیص داده شده."""
        now = datetime.utcnow()
        
        if area_id is None:
            area_id = self._get_default_area_id()
            if area_id is None:
                logger.warning("No area_id available")
                return []
        
        active_track_ids = set()
        results = []
        
        for i, face in enumerate(faces):
            if not face.is_usable:
                logger.debug(f"Face {i} not usable, skipping")
                continue
            
            face_crop = frame[face.y:face.y+face.h, face.x:face.x+face.w]
            if face_crop.size == 0:
                continue
            
            # شناسایی چهره
            identity_type, identity_id, confidence, review_status = (
                self._identify_face(face_crop)
            )
            
            logger.debug(
                f"Face {i}: type={identity_type.value}, "
                f"id={identity_id}, conf={confidence:.3f}"
            )
            
            # تولید track_id
            qx = face.x // 50 * 50
            qy = face.y // 50 * 50
            track_id = f"temp_{camera_id}_{qx}_{qy}"
            
            # مدیریت ناشناس‌ها
            if identity_type == PersonType.UNKNOWN:
                logger.debug(
                    f"Processing unknown face. "
                    f"Track: {track_id}, "
                    f"Unknown service: {self.unknown_service is not None}"
                )
                
                if track_id in self._temp_to_unknown:
                    identity_id = self._temp_to_unknown[track_id]
                    logger.debug(f"Reusing unknown id={identity_id} for track {track_id}")
                elif self.unknown_service:
                    logger.debug(f"Creating new unknown for track {track_id}")
                    new_unknown_id = self.unknown_service.get_or_create_unknown(
                        face_crop=face_crop
                    )
                    if new_unknown_id:
                        identity_id = new_unknown_id
                        self._temp_to_unknown[track_id] = new_unknown_id
                        logger.info(
                            f"✅ Created unknown id={new_unknown_id} for track {track_id}"
                        )
                    else:
                        logger.error(
                            f"❌ Failed to create unknown for track {track_id}"
                        )
                else:
                    logger.warning("Unknown service not available!")
            
            # ساخت label
            if identity_type == PersonType.EMPLOYEE and identity_id:
                label_text = self._get_employee_name(identity_id) or f"EMP-{identity_id}"
            elif identity_type == PersonType.UNKNOWN and identity_id:
                unknown_code = self._get_unknown_code(identity_id)
                label_text = unknown_code or f"UNK-{identity_id}"
            else:
                label_text = "Unknown"
            
            active_track_ids.add(track_id)
            results.append((identity_type, identity_id, label_text))
            
            # بررسی duplicate
            if self._is_duplicate_event(track_id, now):
                logger.debug(f"Duplicate event for track {track_id}, skipping")
                continue
            
            # ایجاد یا به‌روزرسانی جلسه
            self._update_or_create_session(
                camera_id=camera_id,
                area_id=area_id,
                track_id=track_id,
                identity_type=identity_type,
                identity_id=identity_id,
                confidence=confidence,
                review_status=review_status,
                face=face,
                seen_at=now
            )
        
        # بستن جلسات منقضی شده
        self._close_expired_sessions(camera_id, now, active_track_ids)
        
        # پاک کردن mapping های قدیمی
        self._cleanup_temp_mapping(active_track_ids)
        
        return results
    
    def _cleanup_temp_mapping(self, active_track_ids: set):
        """پاک کردن mapping های قدیمی"""
        keys_to_remove = [
            k for k in self._temp_to_unknown
            if k not in active_track_ids
        ]
        for key in keys_to_remove:
            del self._temp_to_unknown[key]
    
    def _get_default_area_id(self) -> Optional[int]:
        try:
            db = SessionLocal()
            try:
                from domain.models import Area
                area = db.query(Area).filter(Area.is_active == True).first()
                return area.id if area else None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error getting default area: {e}")
            return None
    
    def _get_employee_name(self, employee_id: int) -> Optional[str]:
        try:
            db = SessionLocal()
            try:
                emp = db.query(Employee).filter(Employee.id == employee_id).first()
                return emp.full_name if emp else None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error getting employee name: {e}")
            return None
    
    def _get_unknown_code(self, unknown_id: int) -> Optional[str]:
        """گرفتن کد ناشناس"""
        try:
            db = SessionLocal()
            try:
                unknown = db.query(UnknownPerson).filter(
                    UnknownPerson.id == unknown_id
                ).first()
                return unknown.code if unknown else None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error getting unknown code: {e}")
            return None
    
    def _identify_face(
        self, face_crop
    ) -> Tuple[PersonType, Optional[int], float, str]:
        """شناسایی چهره با FaceEngine."""
        try:
            label, confidence = self.face_engine.recognize(face_crop)
            
            logger.debug(
                f"FaceEngine result: label={label}, confidence={confidence:.3f}, "
                f"thresholds: high={self.high_conf_threshold}, low={self.low_conf_threshold}"
            )
            
            if label == self.face_engine.LABEL_UNKNOWN:
                return (
                    PersonType.UNKNOWN,
                    None,
                    confidence,
                    ReviewStatus.AUTO.value
                )
            
            elif label == self.face_engine.LABEL_UNCERTAIN:
                return (
                    PersonType.UNKNOWN,
                    None,
                    confidence,
                    ReviewStatus.PENDING.value
                )
            
            else:
                try:
                    db = SessionLocal()
                    try:
                        employee = db.query(Employee).filter(
                            Employee.id == label
                        ).first()
                        
                        if employee and employee.is_active:
                            return (
                                PersonType.EMPLOYEE,
                                label,
                                confidence,
                                ReviewStatus.AUTO.value
                            )
                        else:
                            return (
                                PersonType.UNKNOWN,
                                None,
                                confidence,
                                ReviewStatus.AUTO.value
                            )
                    finally:
                        db.close()
                except Exception as e:
                    logger.error(f"Error checking employee: {e}")
                    return (
                        PersonType.UNKNOWN,
                        None,
                        confidence,
                        ReviewStatus.AUTO.value
                    )
        
        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (
                PersonType.UNKNOWN,
                None,
                0.0,
                ReviewStatus.AUTO.value
            )
    
    def _is_duplicate_event(self, track_id: str, now: datetime) -> bool:
        with self._lock:
            last_time = self._last_event_time.get(track_id)
            
            if last_time:
                elapsed = (now - last_time).total_seconds()
                if elapsed < self.duplicate_event_interval:
                    return True
            
            self._last_event_time[track_id] = now
            
            if len(self._last_event_time) > 1000:
                cutoff = now - datetime.timedelta(
                    seconds=self.duplicate_event_interval * 2
                )
                self._last_event_time = {
                    k: v for k, v in self._last_event_time.items()
                    if v > cutoff
                }
            
            return False
    
    def _update_or_create_session(
        self,
        camera_id: int,
        area_id: int,
        track_id: str,
        identity_type: PersonType,
        identity_id: Optional[int],
        confidence: float,
        review_status: str,
        face,
        seen_at: datetime
    ) -> None:
        """ایجاد یا به‌روزرسانی جلسه حضور"""
        db = SessionLocal()
        session_id = None
        
        try:
            session = db.query(AttendanceSession).filter(
                AttendanceSession.status == SessionStatus.ACTIVE.value,
                AttendanceSession.camera_id == camera_id,
                AttendanceSession.temp_track_id == track_id
            ).first()
            
            if session:
                session.last_seen_at = seen_at
                
                if (identity_type == PersonType.EMPLOYEE and 
                    session.person_type != PersonType.EMPLOYEE.value):
                    session.person_type = PersonType.EMPLOYEE.value
                    session.employee_id = identity_id
                    session.unknown_id = None
                    logger.info(
                        f"Session {session.id} upgraded to employee {identity_id}"
                    )
                
                if review_status == ReviewStatus.PENDING.value:
                    session.review_status = ReviewStatus.PENDING.value
                
                if session.confidence:
                    session.confidence = (session.confidence + confidence) / 2
                else:
                    session.confidence = confidence
                
                db.commit()
                db.refresh(session)
                session_id = session.id
            else:
                new_session = AttendanceSession(
                    person_type=identity_type.value,
                    employee_id=(
                        identity_id if identity_type == PersonType.EMPLOYEE else None
                    ),
                    unknown_id=(
                        identity_id if identity_type == PersonType.UNKNOWN else None
                    ),
                    temp_track_id=track_id,
                    area_id=area_id,
                    camera_id=camera_id,
                    first_seen_at=seen_at,
                    last_seen_at=seen_at,
                    status=SessionStatus.ACTIVE.value,
                    confidence=confidence,
                    review_status=review_status,
                    shift=self._determine_shift(seen_at).value
                )
                
                db.add(new_session)
                db.commit()
                db.refresh(new_session)
                session_id = new_session.id
                
                person_label = (
                    f"emp_id={identity_id}" if identity_type == PersonType.EMPLOYEE
                    else f"unk_id={identity_id}"
                )
                logger.info(
                    f"✅ New session created: id={new_session.id}, "
                    f"type={identity_type.value}, "
                    f"{person_label}, camera={camera_id}"
                )
            
            sighting = Sighting(
                session_id=session_id,
                camera_id=camera_id,
                area_id=area_id,
                person_type=identity_type.value,
                employee_id=(
                    identity_id if identity_type == PersonType.EMPLOYEE else None
                ),
                unknown_id=(
                    identity_id if identity_type == PersonType.UNKNOWN else None
                ),
                temp_track_id=track_id,
                seen_at=seen_at,
                bbox=f"{face.x},{face.y},{face.w},{face.h}",
                confidence=confidence,
                quality_score=face.quality_score
            )
            
            db.add(sighting)
            db.commit()
        
        except Exception as e:
            logger.error(f"❌ Error in session processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            db.rollback()
        finally:
            db.close()
    
    def _close_expired_sessions(
        self,
        camera_id: int,
        now: datetime,
        active_track_ids: set
    ) -> None:
        """بستن جلسات منقضی شده"""
        db = SessionLocal()
        try:
            active_sessions = db.query(AttendanceSession).filter(
                AttendanceSession.status == SessionStatus.ACTIVE.value,
                AttendanceSession.camera_id == camera_id
            ).all()
            
            closed_count = 0
            for session in active_sessions:
                if session.temp_track_id not in active_track_ids:
                    elapsed = (now - session.last_seen_at).total_seconds()
                    
                    if elapsed >= self.session_close_timeout:
                        session.status = SessionStatus.CLOSED.value
                        session.ended_at = now
                        if session.first_seen_at:
                            session.duration_seconds = int(
                                (session.ended_at - session.first_seen_at).total_seconds()
                            )
                        closed_count += 1
                        logger.info(
                            f"Session closed: id={session.id}, "
                            f"type={session.person_type}, "
                            f"duration={session.duration_seconds}s"
                        )
            
            if closed_count > 0:
                db.commit()
        
        except Exception as e:
            logger.error(f"Error closing expired sessions: {e}")
            db.rollback()
        finally:
            db.close()
    
    def close_all_active_sessions(self) -> int:
        """بستن همه جلسات فعال"""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            active_sessions = db.query(AttendanceSession).filter(
                AttendanceSession.status == SessionStatus.ACTIVE.value
            ).all()
            
            count = 0
            for session in active_sessions:
                session.status = SessionStatus.CLOSED.value
                session.ended_at = now
                if session.first_seen_at:
                    session.duration_seconds = int(
                        (session.ended_at - session.first_seen_at).total_seconds()
                    )
                count += 1
            
            db.commit()
            logger.info(f"Closed {count} active sessions on shutdown")
            return count
        except Exception as e:
            logger.error(f"Error closing all sessions: {e}")
            db.rollback()
            return 0
        finally:
            db.close()
    
    def _determine_shift(self, dt: datetime) -> ShiftType:
        """تعیین شیفت بر اساس ساعت"""
        hour = dt.hour
        
        if 6 <= hour < 14:
            return ShiftType.MORNING
        elif 14 <= hour < 22:
            return ShiftType.AFTERNOON
        else:
            return ShiftType.NIGHT