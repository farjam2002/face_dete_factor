from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from urllib.parse import quote

from infrastructure.database import SessionLocal
from infrastructure.repositories import AreaRepository
from domain.models import Area, Camera, AttendanceSession


router = APIRouter(prefix="/areas", tags=["areas"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def areas_page(
    request: Request,
    error: Optional[str] = None,
    success: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """صفحه مدیریت محدوده‌ها"""
    templates = request.app.state.templates
    areas = AreaRepository.get_all(db)

    area_stats = []
    for area in areas:
        camera_count = db.query(Camera).filter(Camera.area_id == area.id).count()
        area_stats.append({
            "area": area,
            "camera_count": camera_count
        })

    return templates.TemplateResponse(
        request=request,
        name="pages/areas.html",
        context={
            "title": "مدیریت محدوده‌ها",
            "active_page": "areas",
            "area_stats": area_stats,
            "error": error,
            "success": success
        }
    )


@router.post("/", response_class=HTMLResponse)
def create_area(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """افزودن محدوده جدید"""
    name = (name or "").strip()

    if not name:
        return RedirectResponse(
            url="/areas/?error=" + quote("نام محدوده الزامی است."),
            status_code=303
        )

    existing = AreaRepository.get_by_name(db, name)
    if existing:
        return RedirectResponse(
            url="/areas/?error=" + quote(f"محدوده‌ای با نام «{name}» قبلاً وجود دارد."),
            status_code=303
        )

    area = Area(
        name=name,
        description=(description or "").strip() or None,
        is_active=(is_active == "on")
    )

    AreaRepository.create(db, area)

    return RedirectResponse(
        url="/areas/?success=" + quote("محدوده با موفقیت ایجاد شد."),
        status_code=303
    )


@router.post("/{area_id}/edit", response_class=HTMLResponse)
def edit_area(
    area_id: int,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """ویرایش محدوده"""
    area = AreaRepository.get_by_id(db, area_id)
    if not area:
        raise HTTPException(status_code=404, detail="محدوده یافت نشد")

    name = (name or "").strip()

    if not name:
        return RedirectResponse(
            url="/areas/?error=" + quote("نام محدوده الزامی است."),
            status_code=303
        )

    existing = AreaRepository.get_by_name(db, name)
    if existing and existing.id != area_id:
        return RedirectResponse(
            url="/areas/?error=" + quote(f"محدوده‌ای با نام «{name}» قبلاً وجود دارد."),
            status_code=303
        )

    area.name = name
    area.description = (description or "").strip() or None
    area.is_active = (is_active == "on")

    AreaRepository.update(db, area)

    return RedirectResponse(
        url="/areas/?success=" + quote("محدوده با موفقیت ویرایش شد."),
        status_code=303
    )


@router.post("/{area_id}/delete", response_class=HTMLResponse)
def delete_area(area_id: int, db: Session = Depends(get_db)):
    """حذف محدوده"""
    area = AreaRepository.get_by_id(db, area_id)
    if not area:
        raise HTTPException(status_code=404, detail="محدوده یافت نشد")

    camera_count = db.query(Camera).filter(Camera.area_id == area_id).count()
    if camera_count > 0:
        return RedirectResponse(
            url="/areas/?error=" + quote(
                "این محدوده به دوربین متصل است و قابل حذف نیست. ابتدا دوربین‌ها را جدا کنید."
            ),
            status_code=303
        )

    session_count = db.query(AttendanceSession).filter(
        AttendanceSession.area_id == area_id
    ).count()

    if session_count > 0:
        return RedirectResponse(
            url="/areas/?error=" + quote(
                "این محدوده دارای سابقه حضور است و قابل حذف نیست."
            ),
            status_code=303
        )

    db.delete(area)
    db.commit()

    return RedirectResponse(
        url="/areas/?success=" + quote("محدوده حذف شد."),
        status_code=303
    )


@router.post("/{area_id}/toggle", response_class=HTMLResponse)
def toggle_area(area_id: int, db: Session = Depends(get_db)):
    """فعال/غیرفعال کردن محدوده"""
    area = AreaRepository.get_by_id(db, area_id)
    if not area:
        raise HTTPException(status_code=404, detail="محدوده یافت نشد")

    area.is_active = not area.is_active
    AreaRepository.update(db, area)

    return RedirectResponse(url="/areas/", status_code=303)


@router.get("/api/list")
def areas_api_list(db: Session = Depends(get_db)):
    """لیست محدوده‌های فعال برای استفاده در dropdown"""
    areas = db.query(Area).filter(Area.is_active == True).all()
    return [
        {
            "id": a.id,
            "name": a.name,
            "description": a.description
        }
        for a in areas
    ]