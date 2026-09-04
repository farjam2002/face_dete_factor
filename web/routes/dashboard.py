from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from infrastructure.database import SessionLocal
from domain.models import (
    Camera, Employee, UnknownPerson,
    AttendanceSession, Sighting
)
from domain.enums import SessionStatus, PersonType


router = APIRouter(tags=["dashboard"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request, db: Session = Depends(get_db)):
    """صفحه اصلی داشبورد"""
    templates = request.app.state.templates
    
    # آمار کلی
    total_cameras = db.query(Camera).count()
    active_cameras = db.query(Camera).filter(Camera.is_active == True).count()
    
    total_employees = db.query(Employee).count()
    active_employees = db.query(Employee).filter(Employee.is_active == True).count()
    
    total_unknowns = db.query(UnknownPerson).filter(
        UnknownPerson.merged_into_id == None
    ).count()
    
    # جلسات فعال
    active_sessions = db.query(AttendanceSession).filter(
        AttendanceSession.status == SessionStatus.ACTIVE.value
    ).count()
    
    # جلسات امروز
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    sessions_today = db.query(AttendanceSession).filter(
        AttendanceSession.first_seen_at >= today
    ).count()
    
    return templates.TemplateResponse(
        request=request,
        name="pages/dashboard.html",
        context={
            "title": "داشبورد",
            "active_page": "dashboard",
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "total_employees": total_employees,
            "active_employees": active_employees,
            "total_unknowns": total_unknowns,
            "active_sessions": active_sessions,
            "sessions_today": sessions_today
        }
    )


@router.get("/api/dashboard/live")
def get_live_dashboard_data(request: Request, db: Session = Depends(get_db)):
    """دریافت داده‌های زنده داشبورد"""
    
    # آمار کلی
    total_cameras = db.query(Camera).count()
    active_cameras = db.query(Camera).filter(Camera.is_active == True).count()
    
    total_employees = db.query(Employee).count()
    active_employees = db.query(Employee).filter(Employee.is_active == True).count()
    
    total_unknowns = db.query(UnknownPerson).filter(
        UnknownPerson.merged_into_id == None
    ).count()
    
    # جلسات فعال
    active_sessions = db.query(AttendanceSession).filter(
        AttendanceSession.status == SessionStatus.ACTIVE.value
    ).count()
    
    # جلسات امروز
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    sessions_today = db.query(AttendanceSession).filter(
        AttendanceSession.first_seen_at >= today
    ).count()
    
    # حضورهای اخیر (آخرین ۱۰ تای بسته شده)
    recent_sessions = db.query(AttendanceSession).filter(
        AttendanceSession.status == SessionStatus.CLOSED.value
    ).order_by(
        AttendanceSession.ended_at.desc()
    ).limit(10).all()
    
    recent_data = []
    for session in recent_sessions:
        person_name = "—"
        
        if session.person_type == PersonType.EMPLOYEE.value and session.employee_id:
            emp = db.query(Employee).filter(Employee.id == session.employee_id).first()
            if emp:
                person_name = emp.full_name
        elif session.person_type == PersonType.UNKNOWN.value and session.unknown_id:
            unknown = db.query(UnknownPerson).filter(
                UnknownPerson.id == session.unknown_id
            ).first()
            if unknown:
                person_name = unknown.code
        else:
            person_name = "موقت"
        
        recent_data.append({
            "id": session.id,
            "person_type": session.person_type,
            "person_name": person_name,
            "first_seen": session.first_seen_at.strftime("%H:%M:%S") if session.first_seen_at else "—",
            "last_seen": session.last_seen_at.strftime("%H:%M:%S") if session.last_seen_at else "—",
            "duration_seconds": session.duration_seconds or 0,
            "confidence": session.confidence or 0
        })
    
    # دوربین‌های فعال
    camera_service = request.app.state.camera_service
    cameras_data = []
    
    for cam in db.query(Camera).filter(Camera.is_active == True).all():
        source = camera_service.get_source(cam.id) if camera_service else None
        
        cam_info = {
            "id": cam.id,
            "name": cam.name,
            "status": cam.status,
            "area_name": cam.area.name if cam.area else "—",
            "is_running": source is not None and source.is_open()
        }
        
        cameras_data.append(cam_info)
    
    return JSONResponse({
        "stats": {
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "total_employees": total_employees,
            "active_employees": active_employees,
            "total_unknowns": total_unknowns,
            "active_sessions": active_sessions,
            "sessions_today": sessions_today
        },
        "cameras": cameras_data,
        "recent_sessions": recent_data,
        "face_engine_mode": request.app.state.face_engine.mode,
        "face_detector_mode": (
            request.app.state.face_detector.mode 
            if request.app.state.face_detector else "none"
        ),
        "timestamp": datetime.utcnow().isoformat()
    })