from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from infrastructure.database import SessionLocal
from infrastructure.repositories import CameraRepository


router = APIRouter(prefix="/cameras", tags=["camera_stream"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{camera_id}/frame")
def get_camera_frame(
    camera_id: int,
    request: Request,  # <-- این import شده از fastapi
    db: Session = Depends(get_db)
):
    """دریافت آخرین فریم دوربین به صورت JPEG"""
    # بررسی وجود دوربین
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    # دسترسی به CameraService از state اپلیکیشن
    camera_service = getattr(request.app.state, "camera_service", None)
    if not camera_service:
        raise HTTPException(
            status_code=503,
            detail="سرویس دوربین فعال نیست"
        )
    
    frame_bytes = camera_service.get_last_frame_jpg(camera_id)
    
    if not frame_bytes:
        raise HTTPException(
            status_code=404,
            detail="فریمی دریافت نشده است"
        )
    
    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )