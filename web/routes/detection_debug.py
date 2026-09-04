import cv2
import numpy as np
import logging
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import Response, HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from infrastructure.database import SessionLocal
from domain.models import Camera


logger = logging.getLogger("attendance")


router = APIRouter(prefix="/detection-debug", tags=["detection_debug"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def debug_page(request: Request, db: Session = Depends(get_db)):
    """صفحه عیب‌یابی تشخیص چهره"""
    templates = request.app.state.templates
    cameras = db.query(Camera).filter(Camera.is_active == True).all()
    
    return templates.TemplateResponse(
        request=request,
        name="pages/detection_debug.html",
        context={
            "title": "عیب‌یابی تشخیص چهره",
            "active_page": "detection",
            "cameras": cameras
        }
    )


@router.get("/{camera_id}/test")
def test_detection(camera_id: int, request: Request, db: Session = Depends(get_db)):
    """تست تشخیص چهره روی یک فریم"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    camera_service = request.app.state.camera_service
    face_detector = request.app.state.face_detector
    
    source = camera_service.get_source(camera_id)
    if not source:
        raise HTTPException(status_code=404, detail="دوربین در حال اجرا نیست")
    
    frame = source.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="فریمی دریافت نشده است")
    
    # تشخیص چهره‌ها
    faces = face_detector.detect(frame)
    
    # اطلاعات debug
    debug_info = {
        "mode": face_detector.mode,
        "frame_shape": list(frame.shape),
        "faces_count": len(faces),
        "faces": []
    }
    
    for i, face in enumerate(faces):
        debug_info["faces"].append({
            "index": i,
            "x": face.x,
            "y": face.y,
            "w": face.w,
            "h": face.h,
            "confidence": face.confidence,
            "quality_score": face.quality_score,
            "is_usable": face.is_usable
        })
    
    # رسم جعبه‌ها (حتی اگر usable نباشند)
    result_frame = frame.copy()
    
    for face in faces:
        if face.is_usable:
            color = (0, 255, 0)  # سبز
        else:
            color = (0, 165, 255)  # نارنجی
        
        cv2.rectangle(
            result_frame,
            (face.x, face.y),
            (face.x + face.w, face.y + face.h),
            color, 2
        )
        
        label = f"{face.confidence:.2f}"
        cv2.putText(
            result_frame, label,
            (face.x, face.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    
    # اضافه کردن متن حالت
    cv2.putText(
        result_frame, f"Mode: {face_detector.mode.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    
    cv2.putText(
        result_frame, f"Faces: {len(faces)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    
    ok, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    if not ok:
        raise HTTPException(status_code=500, detail="خطا در رمزگذاری تصویر")
    
    return JSONResponse({
        "debug": debug_info,
        "image": buf.tobytes().hex()
    })