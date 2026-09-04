from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from urllib.parse import quote

from infrastructure.database import SessionLocal
from infrastructure.repositories import (
    CameraRepository, AreaRepository,
    CameraStatsRepository
)
from domain.models import Camera, CameraStats, AttendanceSession, Sighting, Alert
from domain.enums import CameraStatus


router = APIRouter(prefix="/cameras", tags=["cameras"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def cameras_page(
    request: Request,
    error: Optional[str] = None,
    success: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """صفحه مدیریت دوربین‌ها"""
    templates = request.app.state.templates

    cameras = db.query(Camera).order_by(Camera.id.desc()).all()

    # تبدیل محدوده‌ها به دیکشنری برای JSON-serializable بودن
    areas_raw = AreaRepository.get_all(db)
    areas = [
        {
            "id": a.id,
            "name": a.name,
            "description": a.description or ""
        }
        for a in areas_raw
    ]

    # ساخت لیست دوربین‌ها با اطلاعات اضافی
    camera_list = []
    for cam in cameras:
        stats = CameraStatsRepository.get_by_camera(db, cam.id)

        camera_list.append({
            "camera": cam,
            "stats": stats,
            "area_name": cam.area.name if cam.area else "— بدون محدوده —"
        })

    return templates.TemplateResponse(
        request=request,
        name="pages/cameras.html",
        context={
            "title": "مدیریت دوربین‌ها",
            "active_page": "cameras",
            "camera_list": camera_list,
            "areas": areas,
            "error": error,
            "success": success
        }
    )


@router.post("/", response_class=HTMLResponse)
def create_camera(
    name: str = Form(...),
    rtsp_url: str = Form(...),
    area_id: Optional[int] = Form(None),
    is_active: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """افزودن دوربین جدید"""
    name = (name or "").strip()
    rtsp_url = (rtsp_url or "").strip()

    if not name:
        return RedirectResponse(
            url="/cameras/?error=" + quote("نام دوربین الزامی است."),
            status_code=303
        )

    if not rtsp_url:
        return RedirectResponse(
            url="/cameras/?error=" + quote("آدرس RTSP الزامی است."),
            status_code=303
        )

    # بررسی تکراری نبودن نام
    existing = db.query(Camera).filter(Camera.name == name).first()
    if existing:
        return RedirectResponse(
            url="/cameras/?error=" + quote(f"دوربینی با نام «{name}» قبلاً وجود دارد."),
            status_code=303
        )

    # بررسی معتبر بودن محدوده
    if area_id:
        area = AreaRepository.get_by_id(db, area_id)
        if not area:
            return RedirectResponse(
                url="/cameras/?error=" + quote("محدوده انتخاب‌شده معتبر نیست."),
                status_code=303
            )
    else:
        area_id = None

    camera = Camera(
        name=name,
        rtsp_url=rtsp_url,
        area_id=area_id,
        is_active=(is_active == "on"),
        status=CameraStatus.OFFLINE.value if (is_active == "on") else CameraStatus.DISABLED.value
    )

    camera = CameraRepository.create(db, camera)

    # ایجاد رکورد آمار برای این دوربین
    stats = CameraStats(camera_id=camera.id)
    CameraStatsRepository.create(db, stats)

    return RedirectResponse(
        url="/cameras/?success=" + quote(f"دوربین «{name}» با موفقیت ایجاد شد."),
        status_code=303
    )


@router.post("/{camera_id}/edit", response_class=HTMLResponse)
def edit_camera(
    camera_id: int,
    name: str = Form(...),
    rtsp_url: str = Form(...),
    area_id: Optional[int] = Form(None),
    is_active: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """ویرایش دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")

    name = (name or "").strip()
    rtsp_url = (rtsp_url or "").strip()

    if not name:
        return RedirectResponse(
            url="/cameras/?error=" + quote("نام دوربین الزامی است."),
            status_code=303
        )

    if not rtsp_url:
        return RedirectResponse(
            url="/cameras/?error=" + quote("آدرس RTSP الزامی است."),
            status_code=303
        )

    existing = db.query(Camera).filter(Camera.name == name, Camera.id != camera_id).first()
    if existing:
        return RedirectResponse(
            url="/cameras/?error=" + quote(f"دوربینی با نام «{name}» قبلاً وجود دارد."),
            status_code=303
        )

    if area_id:
        area = AreaRepository.get_by_id(db, area_id)
        if not area:
            return RedirectResponse(
                url="/cameras/?error=" + quote("محدوده انتخاب‌شده معتبر نیست."),
                status_code=303
            )
        camera.area_id = area_id
    else:
        camera.area_id = None

    camera.name = name
    camera.rtsp_url = rtsp_url

    new_is_active = (is_active == "on")
    if camera.is_active != new_is_active:
        camera.is_active = new_is_active
        if new_is_active:
            camera.status = CameraStatus.OFFLINE.value
        else:
            camera.status = CameraStatus.DISABLED.value

    CameraRepository.update(db, camera)

    return RedirectResponse(
        url="/cameras/?success=" + quote(f"دوربین «{name}» با موفقیت ویرایش شد."),
        status_code=303
    )


@router.post("/{camera_id}/delete", response_class=HTMLResponse)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    """حذف دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")

    session_count = db.query(AttendanceSession).filter(
        AttendanceSession.camera_id == camera_id
    ).count()

    if session_count > 0:
        return RedirectResponse(
            url="/cameras/?error=" + quote(
                "این دوربین دارای سابقه حضور است و قابل حذف نیست."
            ),
            status_code=303
        )

    sighting_count = db.query(Sighting).filter(
        Sighting.camera_id == camera_id
    ).count()

    if sighting_count > 0:
        return RedirectResponse(
            url="/cameras/?error=" + quote(
                "این دوربین دارای سابقه مشاهده است و قابل حذف نیست."
            ),
            status_code=303
        )

    db.query(Alert).filter(Alert.camera_id == camera_id).delete()
    db.delete(camera)
    db.commit()

    return RedirectResponse(
        url="/cameras/?success=" + quote("دوربین حذف شد."),
        status_code=303
    )


@router.post("/{camera_id}/toggle", response_class=HTMLResponse)
def toggle_camera(camera_id: int, db: Session = Depends(get_db)):
    """فعال/غیرفعال کردن دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")

    camera.is_active = not camera.is_active
    if camera.is_active:
        camera.status = CameraStatus.OFFLINE.value
    else:
        camera.status = CameraStatus.DISABLED.value

    CameraRepository.update(db, camera)

    return RedirectResponse(url="/cameras/", status_code=303)


@router.get("/api/list")
def cameras_api_list(db: Session = Depends(get_db)):
    """لیست دوربین‌ها برای استفاده در API"""
    cameras = db.query(Camera).order_by(Camera.name).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "area_id": c.area_id,
            "area_name": c.area.name if c.area else None,
            "is_active": c.is_active,
            "status": c.status
        }
        for c in cameras
    ]