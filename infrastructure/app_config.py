import json
from pathlib import Path


DEFAULT_CONFIG = {
    "app": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "data_dir": "data"
    },
    "storage": {
        "db_file": "db/app.db",
        "images_dir": "images",
        "logs_dir": "logs",
        "backups_dir": "backups"
    },
    "cameras": {
        "frame_queue_size": 30,
        "reconnect_delay_seconds": 5,
        "process_every_n_frames": 3,
        "save_last_frame": True
    },
    "processing": {
        "downscale_width": 640,
        "motion_detection_enabled": True,
        "min_motion_area": 500
    },
    "attendance": {
        "session_close_timeout_seconds": 180,
        "duplicate_event_interval_seconds": 10,
        "unknown_stable_min_frames": 5,
        "reidentification_interval_seconds": 20
    },
    "recognition": {
        "high_confidence_threshold": 0.72,
        "low_confidence_threshold": 0.45,
        "use_onnx_if_available": True
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(base_dir: Path) -> dict:
    config_path = base_dir / "config.json"

    if not config_path.exists():
        config_path.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return DEFAULT_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)

    return _deep_merge(DEFAULT_CONFIG, user_config)


def ensure_dirs(config: dict, base_dir: Path) -> None:
    data_dir = base_dir / config["app"]["data_dir"]

    storage = config["storage"]

    dirs = [
        data_dir,
        data_dir / Path(storage["db_file"]).parent,
        data_dir / storage["images_dir"],
        data_dir / storage["images_dir"] / "employees",
        data_dir / storage["images_dir"] / "unknowns",
        data_dir / storage["images_dir"] / "frames",
        data_dir / storage["logs_dir"],
        data_dir / storage["backups_dir"],
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)