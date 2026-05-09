"""
Vision Guard — YOLOv8 + OpenCV hostel safety monitor.

Major sections are marked with banner comments for navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import time
from ultralytics import YOLO

from detection_config import (
    ALERT_COOLDOWN_SECONDS,
    DISPLAY_ID_MATCH_RADIUS_PX,
    DISPLAY_MIN_TRACK_AREA_FRAC,
    DISPLAY_ID_STALE_SECONDS,
    EVIDENCE_COOLDOWN_SECONDS,
    EVIDENCE_DIR,
    FALL_CONFIRM_SECONDS,
    FALL_WIDTH_HEIGHT_RATIO,
    OVERCROWD_TRIGGER_FRAMES,
    PROXIMITY_MIN_PEOPLE,
    PROXIMITY_RADIUS_FRAC,
    PROXIMITY_RADIUS_PX,
    ROI_POINTS,
)

PROJECT_ROOT = Path(__file__).resolve().parent




class ThreatLevel(Enum):
    """Graduated threat tiers shown on-video and tied to fused alert priority."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def threat_from_alert_state(alerts: "AlertState") -> Tuple[ThreatLevel, str]:
    """
    Map fused alert dominance to threat tier + short human-readable reason.
    Mirrors fuse_alerts() priority so higher tiers always dominate on screen.
    """
    if alerts.top_priority == "fall":
        return ThreatLevel.CRITICAL, "Fall detected"
    if alerts.top_priority == "aggressive":
        return ThreatLevel.HIGH, "Aggressive movement"
    if alerts.top_priority == "overcrowd":
        return ThreatLevel.MEDIUM, "Overcrowding / gathering"
    return ThreatLevel.LOW, "Normal"




@dataclass
class AlertState:
    fall_active: bool = False
    overcrowd_active: bool = False
    aggressive_active: bool = False
    top_priority: str = "normal"


@dataclass
class TrackedPerson:
    track_id: int
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    aggressive: bool = False


def detect_people_with_tracking(model: YOLO, frame: np.ndarray) -> List[TrackedPerson]:
    """Run YOLOv8 tracker with persistent IDs for per-person motion cues."""
    results = model.track(
        frame,
        conf=0.5,
        classes=[0],
        persist=True,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    if boxes.id is None:
        return []

    xyxy = boxes.xyxy.int().cpu().tolist()
    ids = boxes.id.int().cpu().tolist()

    tracked: List[TrackedPerson] = []
    for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        tracked.append(
            TrackedPerson(
                track_id=int(track_id),
                box=(int(x1), int(y1), int(x2), int(y2)),
                center=(int(cx), int(cy)),
            )
        )
    return tracked


def build_roi_polygon(frame_shape: Sequence[int]) -> np.ndarray:
    h, w = frame_shape[:2]
    if ROI_POINTS:
        return np.array(ROI_POINTS, dtype=np.int32)
    return np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.int32)


def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def compute_roi_occupancy(
    person_boxes: List[Tuple[int, int, int, int]], roi_polygon: np.ndarray
) -> int:
    roi_people = 0
    for x1, y1, x2, y2 in person_boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if point_in_polygon((cx, cy), roi_polygon):
            roi_people += 1
    return roi_people


def filter_boxes_in_roi(
    person_boxes: List[Tuple[int, int, int, int]], roi_polygon: np.ndarray
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for box in person_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if point_in_polygon((cx, cy), roi_polygon):
            out.append(box)
    return out


def person_centers(
    person_boxes: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    if not person_boxes:
        return np.zeros((0, 2), dtype=np.float64)
    centers = []
    for x1, y1, x2, y2 in person_boxes:
        centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    return np.asarray(centers, dtype=np.float64)


def max_local_cluster_count(centers: np.ndarray, radius: float) -> int:
    if centers.size == 0:
        return 0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = (diff * diff).sum(axis=2)
    r2 = radius * radius
    counts = (dist_sq <= r2).sum(axis=1)
    return int(counts.max())


def proximity_radius_for_frame(frame_shape: Sequence[int]) -> float:
    h, w = frame_shape[:2]
    if PROXIMITY_RADIUS_PX is not None:
        return float(PROXIMITY_RADIUS_PX)
    return PROXIMITY_RADIUS_FRAC * float(min(w, h))


def evaluate_fall(
    boxes: List[Tuple[int, int, int, int]], fall_start_time: float | None
) -> Tuple[bool, float | None]:
    now = time.time()
    possible_fall = False
    for x1, y1, x2, y2 in boxes:
        width = x2 - x1
        height = y2 - y1
        if height > 0 and width > height * FALL_WIDTH_HEIGHT_RATIO:
            possible_fall = True
            break

    if possible_fall:
        if fall_start_time is None:
            fall_start_time = now
        if now - fall_start_time >= FALL_CONFIRM_SECONDS:
            return True, fall_start_time
        return False, fall_start_time

    return False, None


def fuse_alerts(
    fall_active: bool, overcrowd_active: bool, aggressive_active: bool
) -> AlertState:
    """
    Merge independent detectors into one dominant UI / logging state.
    Fall (CRITICAL) overrides aggressive (HIGH) and overcrowding (MEDIUM).
    """
    if fall_active:
        top = "fall"
    elif aggressive_active:
        top = "aggressive"
    elif overcrowd_active:
        top = "overcrowd"
    else:
        top = "normal"
    return AlertState(
        fall_active=fall_active,
        overcrowd_active=overcrowd_active,
        aggressive_active=aggressive_active,
        top_priority=top,
    )


def update_aggressive_motion_state(
    tracked_people: List[TrackedPerson],
    previous_centers: Dict[int, Tuple[int, int]],
    aggressive_until_by_id: Dict[int, float],
    now: float,
    movement_threshold_px: float,
    hold_seconds: float,
) -> Tuple[List[TrackedPerson], bool]:
    """Mark a person aggressive on large frame-to-frame displacement; hold damps flicker."""
    aggressive_active = False
    active_ids = {person.track_id for person in tracked_people}

    stale_center_ids = [tid for tid in previous_centers if tid not in active_ids]
    for tid in stale_center_ids:
        previous_centers.pop(tid, None)
        aggressive_until_by_id.pop(tid, None)

    for person in tracked_people:
        prev_center = previous_centers.get(person.track_id)
        if prev_center is not None:
            dx = float(person.center[0] - prev_center[0])
            dy = float(person.center[1] - prev_center[1])
            movement_px = float((dx * dx + dy * dy) ** 0.5)
            if movement_px >= movement_threshold_px:
                aggressive_until_by_id[person.track_id] = now + hold_seconds

        person.aggressive = aggressive_until_by_id.get(person.track_id, 0.0) > now
        if person.aggressive:
            aggressive_active = True

        previous_centers[person.track_id] = person.center

    return tracked_people, aggressive_active




@dataclass
class _DisplaySlot:
    """One stable HUD identity: holds last seen kinematics for association."""

    display_num: int
    last_center: Tuple[int, int]
    last_seen: float


class DisplayIdManager:
    """
    Maps each frame's YOLO detections to stable display labels.

    Internal API: ``map_tracks_to_display()`` returns ``track_id -> display_num``.
    Rendering turns ``display_num`` into "Person {n}"; all safety logic keeps using
    ``track_id`` from ``TrackedPerson``.
    """

    def __init__(
        self,
        match_radius_px: float = DISPLAY_ID_MATCH_RADIUS_PX,
        stale_seconds: float = DISPLAY_ID_STALE_SECONDS,
        min_track_area_frac: float | None = DISPLAY_MIN_TRACK_AREA_FRAC,
    ) -> None:
        self._match_radius = float(match_radius_px)
        self._stale_seconds = float(stale_seconds)
        self._min_track_area_frac = (
            float(min_track_area_frac) if min_track_area_frac is not None else None
        )
        self._slots: List[_DisplaySlot] = []

    def _distance_sq(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return dx * dx + dy * dy

    def _bbox_area_px(self, box: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return w * h

    def _track_hud_eligible(self, person: TrackedPerson, frame_shape: Sequence[int]) -> bool:
        if self._min_track_area_frac is None or self._min_track_area_frac <= 0:
            return True
        fh, fw = frame_shape[0], frame_shape[1]
        min_px = fw * fh * self._min_track_area_frac
        return float(self._bbox_area_px(person.box)) >= float(min_px)

    def _smallest_unused_display_num(self) -> int:
        """Reuse low numbers after stale slots prune (avoids Person 2+ forever when 1 vanished)."""
        used = {s.display_num for s in self._slots}
        n = 1
        while n in used:
            n += 1
        return n

    def map_tracks_to_display(
        self,
        tracked_people: List[TrackedPerson],
        now: float,
        frame_shape: Sequence[int],
    ) -> Dict[int, int]:
        """Associate current tracks to stable display numbers for this frame only."""
        r2 = self._match_radius * self._match_radius

        candidates: List[Tuple[float, int, int]] = []
        for person in tracked_people:
            tid = int(person.track_id)
            for si, slot in enumerate(self._slots):
                if now - slot.last_seen > self._stale_seconds:
                    continue
                d2 = self._distance_sq(person.center, slot.last_center)
                if d2 <= r2:
                    candidates.append((d2, tid, si))

        candidates.sort(key=lambda t: t[0])

        assigned_track: set[int] = set()
        assigned_slot_idx: set[int] = set()
        track_to_display: Dict[int, int] = {}

        for _d2, tid, si in candidates:
            if tid in assigned_track or si in assigned_slot_idx:
                continue
            slot = self._slots[si]
            for p in tracked_people:
                if int(p.track_id) == tid:
                    slot.last_center = p.center
                    break
            slot.last_seen = now
            track_to_display[tid] = slot.display_num
            assigned_track.add(tid)
            assigned_slot_idx.add(si)

        newcomers = [
            p for p in tracked_people if int(p.track_id) not in assigned_track
        ]
        newcomers.sort(key=lambda p: self._bbox_area_px(p.box), reverse=True)

        for person in newcomers:
            tid = int(person.track_id)
            if not self._track_hud_eligible(person, frame_shape):
                continue
            num = self._smallest_unused_display_num()
            slot = _DisplaySlot(
                display_num=num,
                last_center=person.center,
                last_seen=now,
            )
            self._slots.append(slot)
            track_to_display[tid] = slot.display_num
            assigned_track.add(tid)

        alive: List[_DisplaySlot] = []
        for slot in self._slots:
            if now - slot.last_seen <= self._stale_seconds:
                alive.append(slot)
        self._slots = alive

        return track_to_display




def ensure_evidence_dir(base: str | Path = EVIDENCE_DIR) -> Path:
    p = Path(base)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)

    # Migrate legacy evidence/ screenshots (older runs) so existing dashboard
    # history URLs like /static/evidence/<name> don't render as broken images.
    legacy = PROJECT_ROOT / "evidence"
    if legacy.exists() and legacy.is_dir():
        try:
            import shutil

            for src in legacy.glob("*.jpg"):
                dst = p / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        except Exception:
            # Best-effort: never break the live pipeline if migration fails.
            pass
    return p


def evidence_filename_for_alerts(alerts: AlertState) -> str:
    """Pick a single label for the file; respects fused dominance (fall > aggressive > crowd)."""
    if alerts.fall_active:
        tag = "fall"
    elif alerts.aggressive_active:
        tag = "aggressive"
    elif alerts.overcrowd_active:
        tag = "overcrowding"
    else:
        tag = "unknown"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"evidence_{ts}_{tag}.jpg"


def maybe_save_evidence_screenshot(
    annotated_bgr: np.ndarray,
    alerts: AlertState,
    last_save_time: float,
    now: float,
    evidence_dir: Path,
) -> Tuple[float, Path | None]:
    """
    Save one JPEG per cooldown when any evidence-worthy condition is active.
    Returns (updated_last_save_time, saved_path_or_none).
    """
    want = alerts.fall_active or alerts.overcrowd_active or alerts.aggressive_active
    if not want:
        return last_save_time, None
    if now - last_save_time < EVIDENCE_COOLDOWN_SECONDS:
        return last_save_time, None

    path = evidence_dir / evidence_filename_for_alerts(alerts)
    cv2.imwrite(str(path), annotated_bgr)
    return now, path




def format_console_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_alert_to_console(
    alerts: AlertState,
    threat: ThreatLevel,
    roi_people_count: int,
    local_cluster_max: int,
) -> None:
    """Structured single-line alert for operators (includes wall-clock time)."""
    print(
        f"[{format_console_timestamp()}] [ALERT] priority={alerts.top_priority} "
        f"threat={threat.value} | ROI_people={roi_people_count} "
        f"local_cluster_max={local_cluster_max}"
    )




def draw_visual_alert_chrome(
    frame: np.ndarray,
    threat: ThreatLevel,
    alert_detail: str,
    alerts: AlertState,
) -> None:
    """
    Thick red frame border + clear status line when not LOW.
    Operates in-place on BGR image.
    """
    if alerts.top_priority == "normal":
        return

    h, w = frame.shape[:2]
    border = max(12, min(h, w) // 55)
    red = (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), red, border)

    banner = f"ALERT  |  THREAT: {threat.value}  —  {alert_detail}"
    cv2.putText(
        frame,
        banner,
        (24, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        red,
        2,
        cv2.LINE_AA,
    )


def render_alerts(
    frame: np.ndarray,
    tracked_people: List[TrackedPerson],
    display_by_track: Dict[int, int],
    roi_polygon: np.ndarray,
    roi_people_count: int,
    local_cluster_max: int,
    proximity_radius_px: float,
    alerts: AlertState,
) -> np.ndarray:
    """
    Draw tracking overlays, ROI summary, legacy status line, threat tier, and alert chrome.

    ``display_by_track`` maps each YOLO ``track_id`` to a stable operator-facing number
    (rendered as "Person {n}"); it must be produced by ``DisplayIdManager`` each frame.
    """
    out = frame.copy()

    cv2.polylines(out, [roi_polygon], True, (255, 200, 0), 2)

    for person in tracked_people:
        x1, y1, x2, y2 = person.box
        color = (0, 255, 0)
        if person.aggressive:
            color = (0, 0, 255)
        elif alerts.fall_active:
            w = x2 - x1
            h = y2 - y1
            if h > 0 and w > h * FALL_WIDTH_HEIGHT_RATIO:
                color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, person.center, 4, color, -1)
        tid = int(person.track_id)
        label = (
            f"Person {display_by_track[tid]}"
            if tid in display_by_track
            else "Detecting..."
        )
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    crowd_ok = local_cluster_max < PROXIMITY_MIN_PEOPLE
    roi_text_color = (0, 255, 0) if crowd_ok else (0, 0, 255)
    rpx = int(round(proximity_radius_px))
    cv2.putText(
        out,
        f"ROI: {roi_people_count} people | Local max: {local_cluster_max} @ {rpx}px",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        roi_text_color,
        2,
    )

    threat, detail = threat_from_alert_state(alerts)
    threat_color = {
        ThreatLevel.LOW: (0, 255, 0),
        ThreatLevel.MEDIUM: (0, 200, 255),
        ThreatLevel.HIGH: (0, 100, 255),
        ThreatLevel.CRITICAL: (0, 0, 255),
    }[threat]
    cv2.putText(
        out,
        f"THREAT LEVEL: {threat.value}  |  {detail}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        threat_color,
        2,
    )

    if alerts.top_priority == "aggressive":
        cv2.putText(
            out,
            "AGGRESSIVE MOVEMENT DETECTED",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
    elif alerts.top_priority == "fall":
        cv2.putText(out, "FALL DETECTED", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 3)
    elif alerts.top_priority == "overcrowd":
        cv2.putText(out, "OVERCROWDING", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(out, "STATUS: NORMAL", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    draw_visual_alert_chrome(out, threat, detail, alerts)

    return out

history = []

def generate_frames():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    fall_start_time: float | None = None
    overcrowd_counter = 0
    last_alert_time = 0.0
    last_aggressive_alert_time = 0.0
    last_evidence_save_time = 0.0
    movement_threshold_px = 80.0
    aggressive_hold_seconds = 0.8
    aggressive_alert_cooldown_seconds = 1.5
    previous_centers: Dict[int, Tuple[int, int]] = {}
    aggressive_until_by_id: Dict[int, float] = {}

    roi_polygon: np.ndarray | None = None
    evidence_path = ensure_evidence_dir()
    display_ids = DisplayIdManager()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            if roi_polygon is None:
                roi_polygon = build_roi_polygon(frame.shape)

            tracked_people = detect_people_with_tracking(model, frame)
            person_boxes = [p.box for p in tracked_people]

            roi_people_count = compute_roi_occupancy(person_boxes, roi_polygon)
            boxes_in_roi = filter_boxes_in_roi(person_boxes, roi_polygon)
            centers = person_centers(boxes_in_roi)
            proximity_r = proximity_radius_for_frame(frame.shape)
            local_cluster_max = max_local_cluster_count(centers, proximity_r)

            if local_cluster_max >= PROXIMITY_MIN_PEOPLE:
                overcrowd_counter += 1
            else:
                overcrowd_counter = 0
            overcrowd_active = overcrowd_counter >= OVERCROWD_TRIGGER_FRAMES

            fall_active, fall_start_time = evaluate_fall(person_boxes, fall_start_time)

            now = time.time()
            tracked_people, aggressive_active = update_aggressive_motion_state(
                tracked_people=tracked_people,
                previous_centers=previous_centers,
                aggressive_until_by_id=aggressive_until_by_id,
                now=now,
                movement_threshold_px=movement_threshold_px,
                hold_seconds=aggressive_hold_seconds,
            )

            display_by_track = display_ids.map_tracks_to_display(
                tracked_people, now, frame.shape
            )

            alerts = fuse_alerts(fall_active, overcrowd_active, aggressive_active)
            threat, _ = threat_from_alert_state(alerts)

            if alerts.top_priority != "normal":
                can_print = now - last_alert_time >= ALERT_COOLDOWN_SECONDS
                if alerts.top_priority == "aggressive":
                    can_print = (
                        can_print
                        and now - last_aggressive_alert_time >= aggressive_alert_cooldown_seconds
                    )
                if can_print:
                    log_alert_to_console(alerts, threat, roi_people_count, local_cluster_max)
                    last_alert_time = now
                    if alerts.top_priority == "aggressive":
                        last_aggressive_alert_time = now

            annotated_frame = render_alerts(
                frame=frame,
                tracked_people=tracked_people,
                display_by_track=display_by_track,
                roi_polygon=roi_polygon,
                roi_people_count=roi_people_count,
                local_cluster_max=local_cluster_max,
                proximity_radius_px=proximity_r,
                alerts=alerts,
            )

            last_evidence_save_time, saved_path = maybe_save_evidence_screenshot(
                annotated_bgr=annotated_frame,
                alerts=alerts,
                last_save_time=last_evidence_save_time,
                now=now,
                evidence_dir=evidence_path,
            )
            if saved_path is not None:
                print(f"[{format_console_timestamp()}] [EVIDENCE] saved -> {saved_path}")
                history.append(
    {
        "url": f"/static/evidence/{saved_path.name}",
        "time": datetime.now().strftime("%H:%M:%S"),
        "score": 85,
        "status": threat.value.lower(),
        "zone": "Hostel Corridor",

        # HUD metrics
        "crowd_density": round(min(local_cluster_max * 5, 25), 1),
        "dwell": 12.5,
        "entry_exit": 8.2,
        "motion": 18.4,
        "time_weight": 1.4,
        "zone_weight": 1.8,
    }
)
                

            ok, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()