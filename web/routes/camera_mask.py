import cv2
import numpy as np
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Tuple

from infrastructure.database import SessionLocal
from infrastructure.repositories import CameraRepository
from infrastructure.mask_utils import create_mask_overlay


router = APIRouter(prefix="/cameras", tags=["camera_mask"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{camera_id}/mask")
def get_camera_mask(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """دریافت نقاط ماسک فعلی دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    mask_points = []
    if camera.mask_points:
        try:
            if isinstance(camera.mask_points, str):
                mask_points = json.loads(camera.mask_points)
            else:
                mask_points = camera.mask_points
        except Exception:
            mask_points = []
    
    return {
        "camera_id": camera_id,
        "mask_points": mask_points,
        "has_mask": len(mask_points) >= 3
    }


@router.post("/{camera_id}/mask")
async def save_camera_mask(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """ذخیره نقاط ماسک دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    try:
        data = await request.json()
        mask_points = data.get("mask_points", [])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON نامعتبر: {e}")
    
    # اعتبارسنجی
    if not isinstance(mask_points, list):
        raise HTTPException(status_code=400, detail="mask_points باید آرایه باشد")
    
    # اعتبارسنجی هر نقطه
    valid_points = []
    for point in mask_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                valid_points.append((x, y))
        except (ValueError, TypeError):
            continue
    
    if 0 < len(valid_points) < 3:
        raise HTTPException(
            status_code=400,
            detail="برای ماسک معتبر حداقل ۳ نقطه لازم است"
        )
    
    # ذخیره در دیتابیس
    camera.mask_points = valid_points if valid_points else None
    db.commit()
    
    # به‌روزرسانی در CameraService در حال اجرا
    camera_service = getattr(request.app.state, "camera_service", None)
    if camera_service:
        camera_service.update_camera_mask(camera_id, valid_points)
    
    return {
        "status": "ok",
        "message": f"ماسک با {len(valid_points)} نقطه ذخیره شد",
        "points_count": len(valid_points)
    }


@router.delete("/{camera_id}/mask")
def delete_camera_mask(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """حذف ماسک دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    camera.mask_points = None
    db.commit()
    
    camera_service = getattr(request.app.state, "camera_service", None)
    if camera_service:
        camera_service.update_camera_mask(camera_id, [])
    
    return {"status": "ok", "message": "ماسک حذف شد"}


@router.get("/{camera_id}/mask/preview")
def preview_camera_mask(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """دریافت تصویر پیش‌نمایش دوربین با overlay ماسک"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    camera_service = getattr(request.app.state, "camera_service", None)
    if not camera_service:
        raise HTTPException(status_code=503, detail="سرویس دوربین فعال نیست")
    
    source = camera_service.get_source(camera_id)
    if not source:
        raise HTTPException(status_code=404, detail="دوربین در حال اجرا نیست")
    
    frame = source.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="فریمی دریافت نشده است")
    
    # بارگذاری ماسک
    mask_points = []
    if camera.mask_points:
        try:
            if isinstance(camera.mask_points, str):
                mask_points = json.loads(camera.mask_points)
            else:
                mask_points = camera.mask_points
        except Exception:
            mask_points = []
    
    # اعمال overlay
    if mask_points and len(mask_points) >= 3:
        frame_with_mask = create_mask_overlay(frame, mask_points)
    else:
        frame_with_mask = frame
    
    ok, buf = cv2.imencode(
        ".jpg", frame_with_mask, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    
    if not ok:
        raise HTTPException(status_code=500, detail="خطا در رمزگذاری تصویر")
    
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/{camera_id}/motion/status")
def get_motion_status(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """دریافت وضعیت تشخیص حرکت دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    camera_service = getattr(request.app.state, "camera_service", None)
    if not camera_service:
        raise HTTPException(status_code=503, detail="سرویس دوربین فعال نیست")
    
    source = camera_service.get_source(camera_id)
    if not source:
        return {
            "camera_id": camera_id,
            "is_running": False,
            "has_motion": False,
            "motion_score": 0.0
        }
    
    return {
        "camera_id": camera_id,
        "is_running": True,
        "has_motion": source.has_motion(),
        "motion_score": source.get_motion_score()
    }