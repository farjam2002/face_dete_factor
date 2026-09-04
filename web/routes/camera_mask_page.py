from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from infrastructure.database import SessionLocal
from infrastructure.repositories import CameraRepository


router = APIRouter(prefix="/cameras", tags=["camera_mask_page"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{camera_id}/mask/edit", response_class=HTMLResponse)
def edit_camera_mask_page(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """صفحه ویرایشگر ماسک دوربین"""
    camera = CameraRepository.get_by_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    templates = request.app.state.templates
    
    return templates.TemplateResponse(
        request=request,
        name="pages/camera_mask.html",
        context={
            "title": f"ویرایش ماسک - {camera.name}",
            "active_page": "cameras",
            "camera": camera
        }
    )