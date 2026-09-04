from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from infrastructure.database import SessionLocal
from domain.models import (
    Camera, AttendanceSession, Sighting, Employee, Area
)
from domain.enums import SessionStatus


router = APIRouter(prefix="/test", tags=["test"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/attendance", response_class=HTMLResponse)
def test_attendance_page(request: Request):
    """صفحه تست جامع حضور و غیاب"""
    templates = request.app.state.templates
    
    return templates.TemplateResponse(
        request=request,
        name="pages/test_attendance.html",
        context={
            "title": "تست حضور و غیاب",
            "active_page": "test"
        }
    )


@router.get("/attendance/status")
def get_attendance_status(request: Request, db: Session = Depends(get_db)):
    """دریافت وضعیت کلی سیستم حضور و غیاب"""
    
    # شمارش‌ها
    total_cameras = db.query(Camera).count()
    active_cameras = db.query(Camera).filter(Camera.is_active == True).count()
    
    total_employees = db.query(Employee).count()
    active_employees = db.query(Employee).filter(Employee.is_active == True).count()
    
    # جلسات فعال
    active_sessions = db.query(AttendanceSession).filter(
        AttendanceSession.status == SessionStatus.ACTIVE.value
    ).all()
    
    # جلسات بسته شده اخیر (۱ ساعت اخیر)
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_closed = db.query(AttendanceSession).filter(
        AttendanceSession.status == SessionStatus.CLOSED.value,
        AttendanceSession.ended_at >= one_hour_ago
    ).order_by(AttendanceSession.ended_at.desc()).limit(10).all()
    
    # sightings اخیر (۱۰ تای آخر)
    recent_sightings = db.query(Sighting).order_by(
        Sighting.seen_at.desc()
    ).limit(10).all()
    
    # اطلاعات دوربین‌های فعال
    camera_service = request.app.state.camera_service
    active_camera_info = []
    
    for cam in db.query(Camera).filter(Camera.is_active == True).all():
        source = camera_service.get_source(cam.id) if camera_service else None
        
        cam_info = {
            "id": cam.id,
            "name": cam.name,
            "status": cam.status,
            "area_name": cam.area.name if cam.area else None,
            "is_running": source is not None and source.is_open(),
            "last_error": cam.last_error
        }
        
        active_camera_info.append(cam_info)
    
    # اطلاعات جلسات فعال
    active_sessions_info = []
    for session in active_sessions:
        employee_name = None
        if session.employee_id:
            emp = db.query(Employee).filter(Employee.id == session.employee_id).first()
            if emp:
                employee_name = emp.full_name
        
        active_sessions_info.append({
            "id": session.id,
            "person_type": session.person_type,
            "employee_name": employee_name,
            "track_id": session.temp_track_id,
            "first_seen": session.first_seen_at.isoformat() if session.first_seen_at else None,
            "last_seen": session.last_seen_at.isoformat() if session.last_seen_at else None,
            "duration_seconds": int(
                (datetime.utcnow() - session.last_seen_at).total_seconds()
            ) if session.last_seen_at else 0,
            "confidence": session.confidence
        })
    
    # اطلاعات sightings اخیر
    sightings_info = []
    for sighting in recent_sightings:
        employee_name = None
        if sighting.employee_id:
            emp = db.query(Employee).filter(Employee.id == sighting.employee_id).first()
            if emp:
                employee_name = emp.full_name
        
        sightings_info.append({
            "id": sighting.id,
            "session_id": sighting.session_id,
            "person_type": sighting.person_type,
            "employee_name": employee_name,
            "seen_at": sighting.seen_at.isoformat() if sighting.seen_at else None,
            "confidence": sighting.confidence,
            "bbox": sighting.bbox
        })
    
    return JSONResponse({
        "summary": {
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "total_employees": total_employees,
            "active_employees": active_employees,
            "active_sessions": len(active_sessions),
            "recent_closed_sessions": len(recent_closed),
            "recent_sightings": len(recent_sightings)
        },
        "cameras": active_camera_info,
        "active_sessions": active_sessions_info,
        "recent_sightings": sightings_info,
        "face_engine_mode": request.app.state.face_engine.mode,
        "face_detector_mode": request.app.state.face_detector.mode if request.app.state.face_detector else "none"
    })


@router.get("/attendance/live-frame/{camera_id}")
def get_live_frame_with_detection(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """گرفتن یک فریم با تشخیص و ثبت حضور"""
    import cv2
    
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    processor_service = request.app.state.processor_service
    camera_service = request.app.state.camera_service
    
    source = camera_service.get_source(camera_id)
    if not source:
        raise HTTPException(status_code=404, detail="دوربین در حال اجرا نیست")
    
    frame = source.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="فریمی دریافت نشده است")
    
    # پردازش با ثبت حضور
    faces, identity_results = processor_service.process_frame(
        camera_id=camera_id,
        area_id=camera.area_id,
        frame=frame,
        register_attendance=True
    )
    
    # رسم نتایج
    result_frame = processor_service.draw_results(frame, faces, identity_results)
    
    ok, buf = cv2.imencode(
        ".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    
    if not ok:
        raise HTTPException(status_code=500, detail="خطا در رمزگذاری تصویر")
    
    # تبدیل به base64
    import base64
    img_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
    return JSONResponse({
        "image": img_base64,
        "faces_count": len(faces),
        "usable_faces": sum(1 for f in faces if f.is_usable),
        "identity_results": [
            {
                "person_type": r[0].value,
                "identity_id": r[1],
                "label": r[2]
            }
            for r in identity_results
        ]
    })