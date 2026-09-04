import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(config: dict, base_dir: Path) -> logging.Logger:
    data_dir = base_dir / config["app"]["data_dir"]
    logs_dir = data_dir / config["storage"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "app.log"

    logger = logging.getLogger("attendance")
    logger.setLevel(logging.INFO)

    # جلوگیری از تکرار هندلر در اجرای مجدد
    if logger.handlers:
        logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger