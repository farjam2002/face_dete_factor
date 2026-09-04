import enum


class PersonType(str, enum.Enum):
    """نوع فرد شناسایی‌شده"""
    EMPLOYEE = "employee"
    UNKNOWN = "unknown"
    TEMPORARY = "temporary"


class SessionStatus(str, enum.Enum):
    """وضعیت جلسه حضور"""
    ACTIVE = "active"
    CLOSED = "closed"


class ReviewStatus(str, enum.Enum):
    """وضعیت بررسی دستی"""
    AUTO = "auto"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CORRECTED = "corrected"


class ShiftType(str, enum.Enum):
    """نوع شیفت کاری"""
    NONE = "none"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"


class CameraStatus(str, enum.Enum):
    """وضعیت دوربین"""
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    ERROR = "error"
    DISABLED = "disabled"


class AlertType(str, enum.Enum):
    """نوع هشدار"""
    CAMERA_DISCONNECTED = "camera_disconnected"
    CAMERA_ERROR = "camera_error"
    UNKNOWN_PERSON = "unknown_person"
    SYSTEM_ERROR = "system_error"
    LOW_CONFIDENCE = "low_confidence"


class AlertLevel(str, enum.Enum):
    """سطح هشدار"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ActionType(str, enum.Enum):
    """نوع عملیات برای لاگ"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    BACKUP = "backup"
    CONFIG_CHANGE = "config_change"