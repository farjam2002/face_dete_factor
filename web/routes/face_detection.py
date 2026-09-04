import cv2
import numpy as np
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import Response, HTMLResponse
from sqlalchemy.orm import Session

from infrastructure.database import SessionLocal
from domain.models import Camera


router = APIRouter(prefix="/detection", tags=["face_detection"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def detection_page(request: Request, db: Session = Depends(get_db)):
    """صفحه تشخیص زنده"""
    templates = request.app.state.templates
    cameras = db.query(Camera).filter(Camera.is_active == True).all()
    
    return templates.TemplateResponse(
        request=request,
        name="pages/detection.html",
        context={
            "title": "تشخیص زنده",
            "active_page": "detection",
            "cameras": cameras
        }
    )


@router.get("/{camera_id}/live")
def detection_live(camera_id: int, request: Request, db: Session = Depends(get_db)):
    """دریافت فریم با تشخیص و شناسایی چهره"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    camera_service = request.app.state.camera_service
    processor_service = request.app.state.processor_service
    
    source = camera_service.get_source(camera_id)
    if not source:
        raise HTTPException(status_code=404, detail="دوربین در حال اجرا نیست")
    
    frame = source.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="فریمی دریافت نشده است")
    
    # پردازش با شناسایی هویت (بدون ثبت مجدد حضور)
    # چون camera_service خودش در حلقه اصلی ثبت می‌کند
    faces, identity_results = processor_service.process_frame(
        camera_id=camera_id,
        area_id=camera.area_id,
        frame=frame,
        register_attendance=False  # فقط شناسایی، ثبت در حلقه اصلی
    )
    
    # رسم نتایج با labels کامل
    result_frame = processor_service.draw_results(frame, faces, identity_results)
    
    ok, buf = cv2.imencode(
        ".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
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