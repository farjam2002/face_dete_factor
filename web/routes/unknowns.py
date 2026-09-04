from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from urllib.parse import quote

from infrastructure.database import SessionLocal
from domain.models import UnknownPerson, Employee, AttendanceSession


router = APIRouter(prefix="/unknowns", tags=["unknowns"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def unknowns_page(
    request: Request,
    error: Optional[str] = None,
    success: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """صفحه مدیریت ناشناس‌ها"""
    templates = request.app.state.templates
    
    # گرفتن ناشناس‌ها (بدون ادغام شده‌ها)
    unknowns = db.query(UnknownPerson).filter(
        UnknownPerson.merged_into_id == None
    ).order_by(UnknownPerson.id.desc()).all()
    
    # گرفتن کارکنان برای انتخاب در فرم اختصاص
    employees = db.query(Employee).filter(
        Employee.is_active == True
    ).order_by(Employee.full_name).all()
    
    # شمارش جلسات هر ناشناس
    unknown_stats = []
    for unknown in unknowns:
        session_count = db.query(AttendanceSession).filter(
            AttendanceSession.unknown_id == unknown.id
        ).count()
        
        unknown_stats.append({
            "unknown": unknown,
            "session_count": session_count
        })
    
    return templates.TemplateResponse(
        request=request,
        name="pages/unknowns.html",
        context={
            "title": "مدیریت ناشناس‌ها",
            "active_page": "unknowns",
            "unknown_stats": unknown_stats,
            "employees": employees,
            "error": error,
            "success": success
        }
    )


@router.post("/{unknown_id}/assign", response_class=HTMLResponse)
def assign_unknown_to_employee(
    unknown_id: int,
    employee_id: int = Form(...),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """اختصاص ناشناس به کارمند"""
    unknown_service = request.app.state.unknown_service
    employee_service = request.app.state.employee_service
    
    success = unknown_service.assign_to_employee(
        unknown_id=unknown_id,
        employee_id=employee_id,
        employee_service=employee_service
    )
    
    if success:
        return RedirectResponse(
            url="/unknowns/?success=" + quote("ناشناس به کارمند اختصاص یافت."),
            status_code=303
        )
    else:
        return RedirectResponse(
            url="/unknowns/?error=" + quote("خطا در اختصاص ناشناس به کارمند."),
            status_code=303
        )


@router.post("/{unknown_id}/merge", response_class=HTMLResponse)
def merge_unknown(
    unknown_id: int,
    target_unknown_id: int = Form(...),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """ادغام ناشناس با ناشناس دیگر"""
    unknown_service = request.app.state.unknown_service
    
    success = unknown_service.merge_unknowns(
        source_unknown_id=unknown_id,
        target_unknown_id=target_unknown_id
    )
    
    if success:
        return RedirectResponse(
            url="/unknowns/?success=" + quote("ناشناس‌ها ادغام شدند."),
            status_code=303
        )
    else:
        return RedirectResponse(
            url="/unknowns/?error=" + quote("خطا در ادغام ناشناس‌ها."),
            status_code=303
        )


@router.post("/{unknown_id}/delete", response_class=HTMLResponse)
def delete_unknown(
    unknown_id: int,
    request: Request = None,
    db: Session = Depends(get_db)
):
    """حذف ناشناس"""
    unknown_service = request.app.state.unknown_service
    
    success = unknown_service.delete_unknown(unknown_id)
    
    if success:
        return RedirectResponse(
            url="/unknowns/?success=" + quote("ناشناس حذف شد."),
            status_code=303
        )
    else:
        return RedirectResponse(
            url="/unknowns/?error=" + quote(
                "خطا در حذف ناشناس. ممکن است جلسات مرتبط داشته باشد."
            ),
            status_code=303
        )


@router.get("/{unknown_id}/image")
def get_unknown_image(
    unknown_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """دریافت تصویر ناشناس"""
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    unknown = db.query(UnknownPerson).filter(
        UnknownPerson.id == unknown_id
    ).first()
    
    if not unknown or not unknown.primary_image_path:
        raise HTTPException(status_code=404, detail="تصویر یافت نشد")
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    config = request.app.state.config
    full_path = base_dir / config["app"]["data_dir"] / unknown.primary_image_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="فایل تصویر یافت نشد")
    
    return FileResponse(
        str(full_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )