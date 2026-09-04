import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from infrastructure.app_config import load_config, ensure_dirs
from infrastructure.logging_setup import setup_logging
from infrastructure.database import init_db
from infrastructure.face_detector import FaceDetector
from infrastructure.face_engine import FaceEngine
from infrastructure.face_gallery import FaceGallery
from services.seed_service import SeedService
from services.camera_service import CameraService
from services.employee_service import EmployeeService
from services.attendance_service import AttendanceService
from services.processor_service import ProcessorService
from services.unknown_service import UnknownService

# Import all routers
from web.routes.dashboard import router as dashboard_router
from web.routes.areas import router as areas_router
from web.routes.cameras import router as cameras_router
from web.routes.camera_stream import router as camera_stream_router
from web.routes.camera_mask import router as camera_mask_router
from web.routes.camera_mask_page import router as camera_mask_page_router
from web.routes.employees import router as employees_router
from web.routes.face_detection import router as face_detection_router
from web.routes.detection_debug import router as detection_debug_router
from web.routes.test_attendance import router as test_attendance_router
from web.routes.unknowns import router as unknowns_router
from web.routes.reports import router as reports_router


BASE_DIR = Path(__file__).resolve().parent

# مسیر مدل تشخیص چهره (ONNX)
FACE_MODEL_PATH = str(BASE_DIR / "models" / "ultraface-RFB-320.onnx")

# بارگذاری اولیه (قبل از ساخت اپ)
_config = load_config(BASE_DIR)
ensure_dirs(_config, BASE_DIR)
_logger = setup_logging(_config, BASE_DIR)
init_db(_config, BASE_DIR)
SeedService.seed_default_areas()

# زیرساخت‌های تشخیص چهره
try:
    _face_detector = FaceDetector(model_path=FACE_MODEL_PATH)
except Exception as e:
    _logger.error(f"Could not initialize face detector: {e}")
    _face_detector = None

_face_engine = FaceEngine(
    high_confidence_threshold=_config["recognition"]["high_confidence_threshold"],
    low_confidence_threshold=_config["recognition"]["low_confidence_threshold"]
)

_images_dir = BASE_DIR / _config["app"]["data_dir"] / _config["storage"]["images_dir"]
_face_gallery = FaceGallery(_images_dir)

_employee_service = EmployeeService(
    base_dir=BASE_DIR,
    config=_config,
    face_gallery=_face_gallery,
    face_engine=_face_engine
)

# سرویس ناشناس‌ها
_unknown_service = UnknownService(
    base_dir=BASE_DIR,
    config=_config
)

# سرویس‌های اصلی
_camera_service = CameraService(_config, BASE_DIR)

_attendance_service = AttendanceService(
    config=_config,
    base_dir=BASE_DIR,
    face_engine=_face_engine,
    unknown_service=_unknown_service
)

_processor_service = ProcessorService(
    face_detector=_face_detector,
    face_engine=_face_engine,
    attendance_service=_attendance_service
)

# اتصال processor به camera service
_camera_service.set_processor_service(_processor_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت چرخه حیات اپلیکیشن"""
    _logger.info("Startup: starting active cameras...")
    _camera_service.start_all()
    
    _logger.info("Startup: building face gallery...")
    _employee_service.rebuild_gallery()
    
    _logger.info("Startup: ready!")
    
    yield
    
    _logger.info("Shutdown: stopping all cameras...")
    _camera_service.stop_all()
    
    _logger.info("Shutdown: closing active sessions...")
    closed = _attendance_service.close_all_active_sessions()
    _logger.info(f"Shutdown complete. Closed {closed} sessions.")


def create_app() -> FastAPI:
    """ساخت و پیکربندی کامل اپلیکیشن FastAPI"""
    
    app = FastAPI(
        title="سیستم حضور و غیاب",
        description="سیستم تشخیص چهره و حضور و غیاب هوشمند",
        version="0.8.0",
        debug=_config["app"]["debug"],
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    
    # ذخیره در state
    app.state.config = _config
    app.state.logger = _logger
    app.state.camera_service = _camera_service
    app.state.face_detector = _face_detector
    app.state.face_engine = _face_engine
    app.state.face_gallery = _face_gallery
    app.state.employee_service = _employee_service
    app.state.attendance_service = _attendance_service
    app.state.processor_service = _processor_service
    app.state.unknown_service = _unknown_service
    
    # قالب‌ها
    templates_dir = BASE_DIR / "web" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates
    
    # فایل‌های استاتیک
    static_dir = BASE_DIR / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # ثبت همه روترها
    app.include_router(dashboard_router)
    app.include_router(areas_router)
    app.include_router(cameras_router)
    app.include_router(camera_stream_router)
    app.include_router(camera_mask_router)
    app.include_router(camera_mask_page_router)
    app.include_router(employees_router)
    app.include_router(face_detection_router)
    app.include_router(detection_debug_router)
    app.include_router(test_attendance_router)
    app.include_router(unknowns_router)
    app.include_router(reports_router)
    
    
    # endpoint های عمومی
    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "message": "سیستم در حال اجرا است.",
            "version": "0.8.0",
            "face_engine_mode": _face_engine.mode,
            "face_detector": _face_detector.mode if _face_detector else "not available"
        }
    
    @app.get("/api/live")
    def live():
        return {
            "status": "ok",
            "cameras": [],
            "alerts": [],
            "stats": {
                "total_cameras": 0,
                "active_cameras": 0,
                "total_employees": 0,
                "total_unknowns": 0,
                "active_sessions": 0
            }
        }
    
    _logger.info("Application started successfully.")
    _logger.info(f"Face engine mode: {_face_engine.mode}")
    _logger.info(f"Face detector: {_face_detector.mode if _face_detector else 'NOT AVAILABLE'}")
    
    return app


# ساخت نمونه اپلیکیشن
app = create_app()


# نقطه شروع اجرا
if __name__ == "__main__":
    app_config = _config["app"]
    
    print("=" * 60)
    print("   سیستم حضور و غیاب")
    print("=" * 60)
    print(f"   آدرس: http://{app_config['host']}:{app_config['port']}")
    print(f"   داشبورد: http://127.0.0.1:{app_config['port']}")
    print(f"   تشخیص زنده: http://127.0.0.1:{app_config['port']}/detection/")
    print(f"   ناشناس‌ها: http://127.0.0.1:{app_config['port']}/unknowns/")
    print(f"   تست حضور: http://127.0.0.1:{app_config['port']}/test/attendance")
    print(f"   مستندات API: http://127.0.0.1:{app_config['port']}/docs")
    print("=" * 60)
    print("   برای توقف سرور: Ctrl + C")
    print("=" * 60)
    print()
    
    uvicorn.run(
        app,
        host=app_config["host"],
        port=int(app_config["port"]),
        reload=False,
        log_level="info"
    )