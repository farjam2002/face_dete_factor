from fastapi import (
    APIRouter, Request, Depends, HTTPException, Form, UploadFile, File
)
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.orm import Session
from typing import Optional
from urllib.parse import quote
from pathlib import Path

from infrastructure.database import SessionLocal
from domain.models import Employee, FaceImage


router = APIRouter(prefix="/employees", tags=["employees"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def employees_page(
    request: Request,
    error: Optional[str] = None,
    success: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """صفحه مدیریت کارکنان"""
    templates = request.app.state.templates
    employee_service = request.app.state.employee_service
    
    employees = db.query(Employee).order_by(Employee.id.desc()).all()
    
    # ساخت لیست با تعداد چهره‌ها
    employee_list = []
    for emp in employees:
        face_count = db.query(FaceImage).filter(
            FaceImage.employee_id == emp.id
        ).count()
        employee_list.append({
            "employee": emp,
            "face_count": face_count
        })
    
    return templates.TemplateResponse(
        request=request,
        name="pages/employees.html",
        context={
            "title": "مدیریت کارکنان",
            "active_page": "employees",
            "employee_list": employee_list,
            "error": error,
            "success": success
        }
    )


@router.post("/", response_class=HTMLResponse)
def create_employee(
    personnel_code: str = Form(...),
    full_name: str = Form(...),
    is_active: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """افزودن کارمند جدید"""
    personnel_code = (personnel_code or "").strip()
    full_name = (full_name or "").strip()
    
    if not personnel_code:
        return RedirectResponse(
            url="/employees/?error=" + quote("کد پرسنلی الزامی است."),
            status_code=303
        )
    
    if not full_name:
        return RedirectResponse(
            url="/employees/?error=" + quote("نام کارمند الزامی است."),
            status_code=303
        )
    
    # بررسی تکراری نبودن کد پرسنلی
    existing = db.query(Employee).filter(
        Employee.personnel_code == personnel_code
    ).first()
    if existing:
        return RedirectResponse(
            url="/employees/?error=" + quote(
                f"کارمندی با کد پرسنلی «{personnel_code}» قبلاً وجود دارد."
            ),
            status_code=303
        )
    
    employee = Employee(
        personnel_code=personnel_code,
        full_name=full_name,
        is_active=(is_active == "on")
    )
    
    db.add(employee)
    db.commit()
    db.refresh(employee)
    
    return RedirectResponse(
        url="/employees/?success=" + quote(f"کارمند «{full_name}» ایجاد شد."),
        status_code=303
    )


@router.post("/{employee_id}/edit", response_class=HTMLResponse)
def edit_employee(
    employee_id: int,
    personnel_code: str = Form(...),
    full_name: str = Form(...),
    is_active: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """ویرایش کارمند"""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="کارمند یافت نشد")
    
    personnel_code = (personnel_code or "").strip()
    full_name = (full_name or "").strip()
    
    if not personnel_code or not full_name:
        return RedirectResponse(
            url="/employees/?error=" + quote("کد پرسنلی و نام الزامی هستند."),
            status_code=303
        )
    
    existing = db.query(Employee).filter(
        Employee.personnel_code == personnel_code,
        Employee.id != employee_id
    ).first()
    if existing:
        return RedirectResponse(
            url="/employees/?error=" + quote(
                f"کد پرسنلی «{personnel_code}» قبلاً استفاده شده است."
            ),
            status_code=303
        )
    
    employee.personnel_code = personnel_code
    employee.full_name = full_name
    employee.is_active = (is_active == "on")
    
    db.commit()
    
    return RedirectResponse(
        url="/employees/?success=" + quote("کارمند ویرایش شد."),
        status_code=303
    )


@router.post("/{employee_id}/delete", response_class=HTMLResponse)
def delete_employee(
    employee_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """حذف کارمند"""
    from domain.models import AttendanceSession
    
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="کارمند یافت نشد")
    
    # بررسی سابقه حضور
    session_count = db.query(AttendanceSession).filter(
        AttendanceSession.employee_id == employee_id
    ).count()
    
    if session_count > 0:
        return RedirectResponse(
            url="/employees/?error=" + quote(
                "این کارمند دارای سابقه حضور است و قابل حذف نیست. "
                "به جای حذف، او را غیرفعال کنید."
            ),
            status_code=303
        )
    
    # حذف چهره‌ها
    employee_service = request.app.state.employee_service
    employee_service.delete_employee_faces(employee_id)
    
    # حذف کارمند
    db.delete(employee)
    db.commit()
    
    return RedirectResponse(
        url="/employees/?success=" + quote("کارمند حذف شد."),
        status_code=303
    )


@router.post("/{employee_id}/toggle", response_class=HTMLResponse)
def toggle_employee(employee_id: int, db: Session = Depends(get_db)):
    """فعال/غیرفعال کردن کارمند"""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="کارمند یافت نشد")
    
    employee.is_active = not employee.is_active
    db.commit()
    
    return RedirectResponse(url="/employees/", status_code=303)


@router.post("/{employee_id}/face/upload", response_class=HTMLResponse)
def upload_face(
    employee_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """آپلود تصویر چهره برای کارمند"""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="کارمند یافت نشد")
    
    # اعتبارسنجی نوع فایل
    if file.content_type and not file.content_type.startswith("image/"):
        return RedirectResponse(
            url="/employees/?error=" + quote("فقط فایل تصویری مجاز است."),
            status_code=303
        )
    
    # خواندن داده
    image_data = file.file.read()
    
    if len(image_data) < 100:
        return RedirectResponse(
            url="/employees/?error=" + quote("فایل تصویر نامعتبر است."),
            status_code=303
        )
    
    # محدودیت اندازه (۵ مگابایت)
    if len(image_data) > 5 * 1024 * 1024:
        return RedirectResponse(
            url="/employees/?error=" + quote("حجم فایل نباید بیش از ۵ مگابایت باشد."),
            status_code=303
        )
    
    employee_service = request.app.state.employee_service
    face_image = employee_service.add_face(employee_id, image_data)
    
    if face_image:
        return RedirectResponse(
            url="/employees/?success=" + quote("چهره با موفقیت ثبت شد."),
            status_code=303
        )
    else:
        return RedirectResponse(
            url="/employees/?error=" + quote("خطا در ذخیره تصویر چهره."),
            status_code=303
        )


@router.post("/face/{face_id}/delete", response_class=HTMLResponse)
def delete_face(
    face_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """حذف یک تصویر چهره"""
    employee_service = request.app.state.employee_service
    success = employee_service.delete_face(face_id)
    
    if success:
        return RedirectResponse(
            url="/employees/?success=" + quote("تصویر چهره حذف شد."),
            status_code=303
        )
    else:
        return RedirectResponse(
            url="/employees/?error=" + quote("خطا در حذف تصویر چهره."),
            status_code=303
        )


@router.get("/face/{face_id}/image")
def get_face_image(
    face_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """دریافت تصویر چهره"""
    from fastapi.responses import FileResponse
    
    face_image = db.query(FaceImage).filter(FaceImage.id == face_id).first()
    if not face_image:
        raise HTTPException(status_code=404, detail="تصویر یافت نشد")
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    config = request.app.state.config
    full_path = base_dir / config["app"]["data_dir"] / face_image.file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="فایل تصویر یافت نشد")
    
    return FileResponse(
        str(full_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )