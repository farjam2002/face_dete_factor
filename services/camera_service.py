import logging
import threading
import time
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from infrastructure.camera_source import CameraSource
from infrastructure.database import SessionLocal
from infrastructure.repositories import (
    CameraRepository, CameraStatsRepository
)
from domain.models import Camera, CameraStats
from domain.enums import CameraStatus


logger = logging.getLogger("attendance")


class CameraService:
    """
    سرویس مدیریت چرخه دوربین‌ها.
    هر دوربین در یک ترد جداگانه اجرا می‌شود.
    """
    
    def __init__(self, config: dict, base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self._sources: Dict[int, CameraSource] = {}
        self._threads: Dict[int, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self._downscale_width = config["processing"]["downscale_width"]
        self._save_frames = config["cameras"]["save_last_frame"]
        self._process_every_n = config["cameras"]["process_every_n_frames"]
        self._motion_enabled = config["processing"]["motion_detection_enabled"]
        self._min_motion_area = config["processing"]["min_motion_area"]
        
        self._frames_dir = (
            base_dir / config["app"]["data_dir"] /
            config["storage"]["images_dir"] / "frames"
        )
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        
        # ProcessorService در app.py تنظیم می‌شود
        self._processor_service = None
    
    def set_processor_service(self, processor_service):
        """تنظیم ProcessorService (از app.py)"""
        self._processor_service = processor_service
        logger.info("CameraService: processor service attached")
    
    def start_all(self):
        """شروع همه دوربین‌های فعال از دیتابیس"""
        db = SessionLocal()
        try:
            cameras = db.query(Camera).filter(Camera.is_active == True).all()
            for cam in cameras:
                self._start_camera_from_model(cam)
        finally:
            db.close()
        
        logger.info(f"CameraService: started {len(self._sources)} cameras")
    
    def _start_camera_from_model(self, cam: Camera):
        """شروع یک دوربین با استفاده از مدل دیتابیس"""
        mask_points = []
        if cam.mask_points:
            try:
                if isinstance(cam.mask_points, str):
                    mask_points = json.loads(cam.mask_points)
                else:
                    mask_points = cam.mask_points
            except Exception as e:
                logger.error(f"Failed to parse mask for camera {cam.id}: {e}")
        
        self.start_camera(
            camera_id=cam.id,
            name=cam.name,
            rtsp_url=cam.rtsp_url,
            area_id=cam.area_id,
            mask_points=mask_points
        )
    
    def stop_all(self):
        """توقف همه دوربین‌ها"""
        self._stop_event.set()
        
        with self._lock:
            for camera_id, source in self._sources.items():
                source.disconnect()
            self._sources.clear()
            self._threads.clear()
        
        logger.info("CameraService: stopped all cameras")
    
    def start_camera(
        self,
        camera_id: int,
        name: str,
        rtsp_url: str,
        area_id: Optional[int] = None,
        mask_points: Optional[List[Tuple[float, float]]] = None
    ):
        """شروع یک دوربین خاص"""
        with self._lock:
            if camera_id in self._sources:
                logger.warning(
                    f"Camera {camera_id} already running, restarting..."
                )
                self._stop_camera_internal(camera_id)
            
            source = CameraSource(
                camera_id=camera_id,
                name=name,
                source=rtsp_url,
                downscale_width=self._downscale_width,
                motion_enabled=self._motion_enabled,
                min_motion_area=self._min_motion_area,
                mask_points=mask_points
            )
            
            thread = threading.Thread(
                target=self._camera_loop,
                args=(source, area_id),
                name=f"cam-{camera_id}",
                daemon=True
            )
            
            self._sources[camera_id] = source
            self._threads[camera_id] = thread
        
        thread.start()
        self._update_camera_status(camera_id, CameraStatus.CONNECTING.value)
    
    def stop_camera(self, camera_id: int):
        """توقف یک دوربین خاص"""
        with self._lock:
            self._stop_camera_internal(camera_id)
        self._update_camera_status(camera_id, CameraStatus.DISABLED.value)
    
    def _stop_camera_internal(self, camera_id: int):
        if camera_id in self._sources:
            source = self._sources.pop(camera_id, None)
            thread = self._threads.pop(camera_id, None)
            
            if source:
                source.disconnect()
            
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
    
    def restart_camera(
        self,
        camera_id: int,
        name: str,
        rtsp_url: str,
        area_id: Optional[int] = None
    ):
        self.stop_camera(camera_id)
        time.sleep(0.5)
        self.start_camera(camera_id, name, rtsp_url, area_id)
    
    def update_camera_mask(
        self,
        camera_id: int,
        mask_points: List[Tuple[float, float]]
    ):
        """به‌روزرسانی ماسک یک دوربین در حال اجرا"""
        with self._lock:
            source = self._sources.get(camera_id)
        
        if source:
            source.update_mask(mask_points)
        else:
            logger.warning(
                f"Camera {camera_id} not running, mask will apply on next start"
            )
    
    def _camera_loop(self, source: CameraSource, area_id: Optional[int]):
        """حلقه اصلی یک دوربین"""
        camera_id = source.camera_id
        reconnect_delay = self.config["cameras"]["reconnect_delay_seconds"]
        frame_counter = 0
        frames_read = 0
        frames_processed = 0
        
        logger.info(f"Camera loop started for '{source.name}' (area_id={area_id})")
        
        while not self._stop_event.is_set():
            with self._lock:
                if camera_id not in self._sources:
                    break
            
            if not source.is_open():
                self._update_camera_status(
                    camera_id, CameraStatus.CONNECTING.value
                )
                
                if not source.connect():
                    error = source.get_last_error() or "Connection failed"
                    self._update_camera_status(
                        camera_id, CameraStatus.ERROR.value, error=error
                    )
                    
                    for _ in range(reconnect_delay * 10):
                        if self._stop_event.is_set():
                            break
                        time.sleep(0.1)
                    continue
            
            frame = source.read_frame()
            
            if frame is None:
                error = source.get_last_error() or "Frame read failed"
                self._update_camera_status(
                    camera_id, CameraStatus.ERROR.value, error=error
                )
                time.sleep(0.5)
                continue
            
            self._update_camera_status(camera_id, CameraStatus.ONLINE.value)
            
            frames_read += 1
            frame_counter += 1
            
            # پردازش هر N فریم
            if frame_counter >= self._process_every_n:
                frames_processed += 1
                frame_counter = 0
                
                # بررسی حرکت (اگر فعال باشد)
                should_process = True
                if self._motion_enabled and not source.has_motion():
                    should_process = False
                
                # پردازش با ProcessorService
                if should_process and self._processor_service:
                    try:
                        self._processor_service.process_frame(
                            camera_id=camera_id,
                            area_id=area_id,
                            frame=frame,
                            register_attendance=True
                        )
                    except Exception as e:
                        logger.error(
                            f"Camera {camera_id} processing error: {e}"
                        )
                
                # به‌روزرسانی آمار
                self._update_camera_stats(
                    camera_id,
                    frames_read,
                    frames_processed
                )
                
                # ذخیره آخرین فریم
                if self._save_frames:
                    frame_path = self._frames_dir / f"cam_{camera_id}_last.jpg"
                    source.save_last_frame(frame_path)
            
            time.sleep(0.01)
        
        source.disconnect()
        logger.info(f"Camera loop stopped for '{source.name}'")
    
    def _update_camera_status(
        self,
        camera_id: int,
        status: str,
        error: Optional[str] = None
    ):
        try:
            db = SessionLocal()
            try:
                CameraRepository.update_status(db, camera_id, status, error)
                
                if status == CameraStatus.ONLINE.value:
                    camera = CameraRepository.get_by_id(db, camera_id)
                    if camera:
                        camera.last_success_at = datetime.utcnow()
                        db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update camera status: {e}")
    
    def _update_camera_stats(
        self,
        camera_id: int,
        frames_read: int,
        frames_processed: int
    ):
        try:
            db = SessionLocal()
            try:
                stats = CameraStatsRepository.get_by_camera(db, camera_id)
                if stats:
                    stats.frames_read = frames_read
                    stats.frames_processed = frames_processed
                    CameraStatsRepository.update(db, stats)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update camera stats: {e}")
    
    def get_last_frame_jpg(self, camera_id: int) -> Optional[bytes]:
        with self._lock:
            source = self._sources.get(camera_id)
        
        if not source:
            return None
        
        return source.get_last_frame_jpg()
    
    def get_source(self, camera_id: int) -> Optional[CameraSource]:
        with self._lock:
            return self._sources.get(camera_id)