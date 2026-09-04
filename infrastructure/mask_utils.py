import cv2
import numpy as np
from typing import List, Tuple, Optional


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> bool:
    """
    بررسی اینکه آیا یک نقطه داخل polygon است یا خیر.
    با الگوریتم Ray Casting.
    
    Args:
        point: (x, y) نقطه مورد نظر
        polygon: لیستی از نقاط polygon
    
    Returns:
        True اگر نقطه داخل polygon باشد
    """
    if not polygon or len(polygon) < 3:
        return True  # اگر ماسک تعریف نشده، همه نقاط معتبر هستند
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


def bbox_inside_mask(
    bbox: Tuple[int, int, int, int],
    polygon: List[Tuple[float, float]],
    min_overlap_ratio: float = 0.5
) -> bool:
    """
    بررسی اینکه آیا bounding box (x, y, w, h) داخل ماسک است یا خیر.
    
    Args:
        bbox: (x, y, width, height) جعبه چهره
        polygon: نقاط ماسک به صورت نرمال‌شده (0 تا 1)
        min_overlap_ratio: حداقل نسبت همپوشانی برای پذیرش
    
    Returns:
        True اگر جعبه داخل ماسک باشد
    """
    if not polygon or len(polygon) < 3:
        return True  # بدون ماسک = همه معتبر
    
    x, y, w, h = bbox
    
    # نمونه‌گیری از نقاط کلیدی جعبه
    test_points = [
        (x, y),              # گوشه بالا-چپ
        (x + w, y),          # گوشه بالا-راست
        (x, y + h),          # گوشه پایین-چپ
        (x + w, y + h),      # گوشه پایین-راست
        (x + w / 2, y + h / 2),  # مرکز
        (x + w / 4, y + h / 4),
        (x + 3 * w / 4, y + h / 4),
        (x + w / 4, y + 3 * h / 4),
        (x + 3 * w / 4, y + 3 * h / 4),
    ]
    
    inside_count = sum(1 for p in test_points if point_in_polygon(p, polygon))
    ratio = inside_count / len(test_points)
    
    return ratio >= min_overlap_ratio


def create_mask_overlay(
    frame: np.ndarray,
    polygon: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3,
    thickness: int = 2
) -> np.ndarray:
    """
    رسم ماسک روی فریم (برای پیش‌نمایش).
    
    Args:
        frame: فریم اصلی
        polygon: نقاط ماسک (نرمال‌شده 0 تا 1 یا پیکسلی)
        color: رنگ مرز (BGR)
        alpha: شفافیت ناحیه داخل ماسک
        thickness: ضخامت خط
    
    Returns:
        فریم با overlay ماسک
    """
    if not polygon or len(polygon) < 3:
        return frame
    
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # تبدیل نقاط نرمال‌شده به پیکسلی
    pts = []
    for px, py in polygon:
        if 0 <= px <= 1 and 0 <= py <= 1:
            # نرمال‌شده است
            pts.append((int(px * w), int(py * h)))
        else:
            # پیکسلی است
            pts.append((int(px), int(py)))
    
    pts_array = np.array(pts, dtype=np.int32)
    
    # پر کردن ناحیه داخل با رنگ شفاف
    cv2.fillPoly(overlay, [pts_array], color)
    
    # ترکیب با فریم اصلی
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # رسم مرز
    cv2.polylines(
        result, [pts_array], isClosed=True,
        color=color, thickness=thickness
    )
    
    # رسم نقاط
    for pt in pts:
        cv2.circle(result, pt, 4, color, -1)
    
    return result


def normalize_polygon(
    polygon: List[Tuple[float, float]],
    frame_width: int,
    frame_height: int
) -> List[Tuple[float, float]]:
    """
    تبدیل نقاط پیکسلی به نرمال‌شده (0 تا 1).
    این کار باعث می‌شود ماسک مستقل از اندازه فریم باشد.
    """
    if not polygon:
        return []
    
    normalized = []
    for x, y in polygon:
        nx = x / frame_width if frame_width > 0 else 0
        ny = y / frame_height if frame_height > 0 else 0
        normalized.append((nx, ny))
    
    return normalized


def denormalize_polygon(
    polygon: List[Tuple[float, float]],
    frame_width: int,
    frame_height: int
) -> List[Tuple[int, int]]:
    """
    تبدیل نقاط نرمال‌شده به پیکسلی.
    """
    if not polygon:
        return []
    
    denormalized = []
    for nx, ny in polygon:
        x = int(nx * frame_width)
        y = int(ny * frame_height)
        denormalized.append((x, y))
    
    return denormalized


def apply_mask_to_frame(
    frame: np.ndarray,
    polygon: List[Tuple[float, float]]
) -> np.ndarray:
    """
    سیاه کردن نواحی خارج از ماسک.
    مفید برای کاهش نویز و تمرکز روی ناحیه مورد نظر.
    """
    if not polygon or len(polygon) < 3:
        return frame
    
    h, w = frame.shape[:2]
    
    # تبدیل نقاط به پیکسلی
    pts = []
    for px, py in polygon:
        if 0 <= px <= 1 and 0 <= py <= 1:
            pts.append((int(px * w), int(py * h)))
        else:
            pts.append((int(px), int(py)))
    
    pts_array = np.array(pts, dtype=np.int32)
    
    # ساخت ماسک سیاه
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_array], 255)
    
    # اعمال ماسک روی فریم
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result