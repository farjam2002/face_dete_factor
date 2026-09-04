from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from infrastructure.database import SessionLocal
from domain.models import (
    Camera, Employee, UnknownPerson,
    AttendanceSession, Sighting
)
from domain.enums import SessionStatus, PersonType


router = APIRouter(prefix="/reports", tags=["reports"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def reports_page(request: Request, db: Session = Depends(get_db)):
    """صفحه گزارش‌ها"""
    templates = request.app.state.templates
    
    # کارکنان برای فیلتر
    employees = db.query(Employee).order_by(Employee.full_name).all()
    
    # دوربین‌ها برای فیلتر
    cameras = db.query(Camera).all()
    
    return templates.TemplateResponse(
        request=request,
        name="pages/reports.html",
        context={
            "title": "گزارش‌ها",
            "active_page": "reports",
            "employees": employees,
            "cameras": cameras
        }
    )


@router.get("/sessions")
def get_sessions_report(
    request: Request,
    start_date: str = None,
    end_date: str = None,
    employee_id: int = None,
    camera_id: int = None,
    person_type: str = None,
    status: str = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """دریافت گزارش جلسات حضور با فیلتر"""
    
    query = db.query(AttendanceSession)
    
    # فیلتر بازه زمانی
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(AttendanceSession.first_seen_at >= start_dt)
        except ValueError:
            pass
    
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(AttendanceSession.first_seen_at < end_dt)
        except ValueError:
            pass
    
    # فیلتر کارمند
    if employee_id:
        query = query.filter(AttendanceSession.employee_id == employee_id)
    
    # فیلتر دوربین
    if camera_id:
        query = query.filter(AttendanceSession.camera_id == camera_id)
    
    # فیلتر نوع فرد
    if person_type:
        query = query.filter(AttendanceSession.person_type == person_type)
    
    # فیلتر وضعیت
    if status:
        query = query.filter(AttendanceSession.status == status)
    
    # مرتب‌سازی و محدودیت
    sessions = query.order_by(
        AttendanceSession.first_seen_at.desc()
    ).limit(limit).all()
    
    # ساخت خروجی
    results = []
    for session in sessions:
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
        
        results.append({
            "id": session.id,
            "person_type": session.person_type,
            "person_name": person_name,
            "first_seen": session.first_seen_at.isoformat() if session.first_seen_at else None,
            "last_seen": session.last_seen_at.isoformat() if session.last_seen_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_seconds": session.duration_seconds or 0,
            "status": session.status,
            "confidence": session.confidence or 0,
            "review_status": session.review_status,
            "camera_id": session.camera_id,
            "area_id": session.area_id
        })
    
    return JSONResponse({
        "count": len(results),
        "sessions": results
    })


@router.get("/employees")
def get_employees_report(
    request: Request,
    db: Session = Depends(get_db)
):
    """گزارش حضور کارکنان"""
    
    employees = db.query(Employee).filter(Employee.is_active == True).all()
    
    results = []
    for emp in employees:
        # جلسات کارمند
        sessions = db.query(AttendanceSession).filter(
            AttendanceSession.employee_id == emp.id
        ).order_by(AttendanceSession.first_seen_at.desc()).limit(10).all()
        
        total_duration = sum(s.duration_seconds or 0 for s in sessions)
        
        results.append({
            "employee_id": emp.id,
            "personnel_code": emp.personnel_code,
            "full_name": emp.full_name,
            "total_sessions": len(sessions),
            "total_duration_seconds": total_duration,
            "sessions": [
                {
                    "id": s.id,
                    "first_seen": s.first_seen_at.isoformat() if s.first_seen_at else None,
                    "duration_seconds": s.duration_seconds or 0
                }
                for s in sessions
            ]
        })
    
    return JSONResponse({
        "count": len(results),
        "employees": results
    })