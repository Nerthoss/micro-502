import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

class State(Enum):
    TAKEOFF         = auto()
    SEARCHING       = auto()   
    FLY_TO_GATE     = auto()
    PASSING_GATE    = auto()
    PASSED_GATE     = auto()

@dataclass
class RawDetection:
    detected: bool
    center: Optional[Tuple[int, int]] = None                 # (u, v) pixel coordinates
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None    # (x, y, w, h) bounding box
    corners: Optional[Tuple[Tuple[int, int], ...]] = None    # (top-left, top-right, bottom-right, bottom-left)

    @property
    def bbox_height(self) -> Optional[float]:
        """Height of the bounding box in pixels, extracted from bbox_xywh."""
        if self.bbox_xywh is None:
            return None
        _, _, _, h = self.bbox_xywh
        return float(h)

    @property
    def bbox_width(self) -> Optional[float]:
        """Width of the bounding box in pixels, extracted from bbox_xywh."""
        if self.bbox_xywh is None:
            return None
        _, _, w, _ = self.bbox_xywh
        return float(w)
    
    @property
    def corners_array(self) -> Optional[np.ndarray]:
        """Corners as a (4, 2) float32 array for use with cv2.solvePnP."""
        if self.corners is None or len(self.corners) != 4:
            return None
        return np.array(self.corners, dtype=np.float32)
    
@dataclass
class GateEstimate:
    position:   np.ndarray  # (x, y, z) in world frame
    yaw:        float       # gate normal yaw in world frame (radians)
    yaw_std:    float       # circular std — large = unreliable
    method:     str         # 'solvepnp' | 'triangulation' | 'aspect_ratio'
    num_frames: int         # frames that contributed

@dataclass
class Setpoint:
    x: float
    y: float
    z: float
    yaw: float

    def as_array(self):
        return np.array([self.x, self.y, self.z, self.yaw])
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
        yield self.yaw
    
@dataclass
class CameraSpecs:
    fov_x: float = 1.5  # radians
    fov_y: float = 1.5  # radians
    img_width: int = 300
    img_height: int = 300

class MyAssignment:
    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.cam_specs = CameraSpecs()
        self.fsm = DroneFSM(specs=self.cam_specs, gate_count=5)

        

    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        camera_frame = camera_data
        control_command = self.fsm.step(sensor_data, camera_frame)
        print(f"  State: {self.fsm.state.name:15s} | "
              f"Setpoint: x={control_command.x:.2f}  y={control_command.y:.2f}  "
              f"z={control_command.z:.2f}  yaw={control_command.yaw:.1f}°"
              f" |  Gate detected: {self.fsm.detector.detect(camera_frame).detected}")
    
        return list(control_command)
    
class GateDetector:
    HSV_LOWER = np.array([140, 50, 120])
    HSV_UPPER = np.array([160, 255, 255])

    GATE_REAL_HEIGHT = 0.4  # meters

    MIN_CONTOUR_AREA = 200.0  # pixels


    def __init__(self, specs: CameraSpecs):
        self.specs = specs

    def detect(self, frame) -> RawDetection:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)    
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.HSV_LOWER, self.HSV_UPPER)

        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return RawDetection(detected=False)

        best = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        area = w * h

        if area < self.MIN_CONTOUR_AREA:
            return RawDetection(detected=False)
        
        center_px = (x + w / 2, y + h / 2)

        # Approximate contour to polygon and extract corners if 4 vertices found
        epsilon = 0.02 * cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, epsilon, True)
        corners = tuple(tuple(pt[0]) for pt in approx) if len(approx) == 4 else None
        
        return RawDetection(detected=True, center=center_px, bbox_xywh=(x, y, w, h), corners=corners)
    
    def estimate_world_position(self, detection: RawDetection, drone_state) -> np.ndarray:
        """
        Estimate gate center position in world frame using back-projection.

        Camera convention (OpenCV):
            x_cam → right
            y_cam → down
            z_cam → forward (optical axis)

        World convention:
            x, y → ground plane
            z    → up

        The camera is assumed to be forward-facing with no pitch/roll,
        so the mapping is:
            x_cam  →  x_world (via yaw rotation)
            y_cam  → -z_world  (down in camera = down in world)
            z_cam  →  forward in world (via yaw rotation)
        """
        cam = self.specs
        u, v = detection.center

        # ── 1. Focal length in pixels ────────────────────────────────────────────
        # fov_x is the full horizontal field of view in radians
        f_px = (cam.img_width / 2.0) / np.tan(cam.fov_x / 2.0)

        # ── 2. Distance estimate from known gate height ──────────────────────────
        # Use bounding-box HEIGHT (pixels) because gate height is fixed at 0.4 m.
        # Gate width is variable (0.3–0.5 m), so don't rely on it.
        bbox_height_px = max(detection.bbox_height, 1.0)   # guard against zero
        distance = f_px * self.GATE_REAL_HEIGHT / bbox_height_px

        # ── 3. Build the ray in camera frame ────────────────────────────────────
        # Normalised image-plane coordinates (OpenCV origin = top-left)
        x_cam = (u - cam.img_width  / 2.0) / f_px   # positive → right
        y_cam = (v - cam.img_height / 2.0) / f_px   # positive → down

        # Ray in camera frame, un-normalised; z_cam = 1 (forward)
        ray_cam = np.array([x_cam, y_cam, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # ── 4. Rotate ray into world frame ───────────────────────────────────────
        # Camera axes mapped to world axes (no pitch / roll assumed):
        #   z_cam (forward) → [cos(yaw), sin(yaw), 0]   (forward on ground plane)
        #   x_cam (right)   → [sin(yaw),-cos(yaw), 0]   (right on ground plane) 
        #   y_cam (down)    → [0,        0,        -1]   (down  → -z_world)
        yaw = drone_state['yaw']
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Columns are world-frame vectors for [x_cam, y_cam, z_cam]
        R_cam_to_world = np.array([
            # x_cam col    y_cam col   z_cam col
            [ sy,          0,          cy ],   # world x
            [-cy,          0,          sy ],   # world y
            [ 0,          -1,          0  ],   # world z
        ])

        ray_world = R_cam_to_world @ ray_cam

        # ── 5. Project along the ray from the drone position ────────────────────
        drone_pos = np.array([
            drone_state['x_global'],
            drone_state['y_global'],
            drone_state['z_global']
        ])

        gate_world = drone_pos + distance * ray_world
        print(f"[GateDetector]  Detected gate at world pos {gate_world}")
        return gate_world  # (x, y, z) in world frame
    
# ──────────────────────────────────────────────────────────────────────────────
#  GateDetection  —  stateful per-gate tracker + detector result
# ──────────────────────────────────────────────────────────────────────────────

class GateDetection:
    """
    Holds the per-frame detection result AND owns the multi-frame history
    needed to estimate gate position and yaw in the world frame.

    Typical usage
    -------------
    detection = GateDetection(cam_specs)          # once, per gate target

    # each frame:
    raw = detector.detect(frame)                  # fills detected, center, etc.
    detection.update_from(raw)                    # copy frame data in
    estimate = detection.estimate(drone_state)    # returns GateEstimate | None
    """

    GATE_REAL_HEIGHT = 0.4   # m — fixed
    GATE_REAL_WIDTH  = 0.4   # m — nominal, used for aspect-ratio yaw only
    MIN_BASELINE     = 0.15  # m — minimum drone displacement for triangulation

    def __init__(self, cam_specs, history_len: int = 20):
        self.cam = cam_specs

        # ── Per-frame detection fields ────────────────────────────────────────
        self.detected:  bool                               = False
        self.center:    Optional[Tuple[float, float]]      = None  # (u, v) pixels
        self.bbox_xywh: Optional[Tuple[int,int,int,int]]   = None  # (x, y, w, h)
        self.corners:   Optional[Tuple[Tuple[int,int],...]] = None  # TL,TR,BR,BL
        self.world_pos: Optional[Tuple[float,float,float]] = None  # last estimate

        # ── Multi-frame history ───────────────────────────────────────────────
        # Each entry: (drone_pos np.ndarray(3,), ray np.ndarray(3,), snapshot)
        # snapshot is a lightweight dict so we don't keep a reference cycle
        self._history: deque = deque(maxlen=history_len)

        # ── Camera intrinsics (computed once) ─────────────────────────────────
        self._K    = self._build_K()
        self._dist = np.zeros(5, dtype=np.float64)  # no distortion in simulation

        # Gate 3-D corners in gate-local frame (origin = gate centre).
        # Gate lies in the XZ plane of its own frame; normal points along +Y.
        # Order: TL, TR, BR, BL  — must match corner detection output.
        H = self.GATE_REAL_HEIGHT / 2.0
        W = self.GATE_REAL_WIDTH  / 2.0
        self._gate_obj_pts = np.array([
            [-W,  H, 0],
            [ W,  H, 0],
            [ W, -H, 0],
            [-W, -H, 0],
        ], dtype=np.float32)

    # ── Properties derived from bbox_xywh ────────────────────────────────────

    @property
    def bbox_height(self) -> Optional[float]:
        return float(self.bbox_xywh[3]) if self.bbox_xywh is not None else None

    @property
    def bbox_width(self) -> Optional[float]:
        return float(self.bbox_xywh[2]) if self.bbox_xywh is not None else None

    @property
    def corners_array(self) -> Optional[np.ndarray]:
        """Corners as (4, 2) float32 for cv2.solvePnP."""
        if self.corners is None or len(self.corners) != 4:
            return None
        return np.array(self.corners, dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_from(self, raw: 'GateDetection') -> None:
        """
        Copy per-frame fields from a freshly built raw detection
        (e.g. the return value of GateDetector.detect()).
        Does NOT touch history or world_pos.
        """
        self.detected  = raw.detected
        self.center    = raw.center
        self.bbox_xywh = raw.bbox_xywh
        self.corners   = raw.corners

    def estimate(self, drone_state: dict) -> Optional[GateEstimate]:
        """
        Run the estimation pipeline for the current frame.
        Returns a GateEstimate, or None if not detected.
        Also updates self.world_pos for convenience.

        Priority:
          1. solvePnP      (corners available)
          2. Triangulation (sufficient drone displacement in history)
          3. Aspect ratio  (single-frame fallback)
        """
        if not self.detected:
            return None

        drone_pos = np.array([
            drone_state['x_global'],
            drone_state['y_global'],
            drone_state['z_global'],
        ])
        ray = self._detection_ray(drone_state)

        # Store a lightweight snapshot so history has no reference cycles
        self._history.append((drone_pos, ray, {
            'bbox_width':  self.bbox_width,
            'bbox_height': self.bbox_height,
        }))

        # ── 1. solvePnP ───────────────────────────────────────────────────────
        if self.corners_array is not None:
            result = self._estimate_solvepnp(drone_state, drone_pos)
            if result is not None:
                self.world_pos = tuple(result.position)
                return result

        # ── 2. Triangulation ──────────────────────────────────────────────────
        result = self._estimate_triangulation()
        if result is not None:
            self.world_pos = tuple(result.position)
            return result

        # ── 3. Aspect-ratio fallback ──────────────────────────────────────────
        result = self._estimate_aspect_ratio(drone_pos, ray)
        self.world_pos = tuple(result.position)
        return result

    def reset(self) -> None:
        """Clear history — call when locking onto a new gate."""
        self._history.clear()
        self.world_pos = None

    # ── Strategy 1 — solvePnP ────────────────────────────────────────────────

    def _estimate_solvepnp(
        self, drone_state: dict, drone_pos: np.ndarray
    ) -> Optional[GateEstimate]:
        ok, rvec, tvec = cv2.solvePnP(
            self._gate_obj_pts,
            self.corners_array,
            self._K,
            self._dist,
            flags=cv2.SOLVEPNP_IPPE,   # best solver for planar targets
        )
        if not ok:
            return None

        R_c2w = self._R_cam_to_world(drone_state)

        # Gate centre in world frame
        pos_world = drone_pos + R_c2w @ tvec.flatten()

        # Gate normal: local +Y axis of gate frame, expressed in camera frame,
        # then rotated to world frame
        R_gate_in_cam, _ = cv2.Rodrigues(rvec)
        normal_world = R_c2w @ R_gate_in_cam[:, 1]
        gate_yaw     = float(np.arctan2(normal_world[1], normal_world[0]))

        return GateEstimate(
            position   = pos_world,
            yaw        = gate_yaw,
            yaw_std    = 0.05,
            method     = 'solvepnp',
            num_frames = 1,
        )

    # ── Strategy 2 — Triangulation ───────────────────────────────────────────

    def _estimate_triangulation(self) -> Optional[GateEstimate]:
        latest_pos = self._history[-1][0]
        usable = [
            (pos, ray, snap) for pos, ray, snap in self._history
            if np.linalg.norm(pos - latest_pos) >= self.MIN_BASELINE
        ]
        if not usable:
            return None
        usable.append(self._history[-1])

        # Least-squares ray intersection:
        #   minimise Σ ||( I − d_i d_iᵀ )(P − o_i)||²
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for pos, ray, _ in usable:
            d  = ray / np.linalg.norm(ray)
            Pi = np.eye(3) - np.outer(d, d)
            A += Pi
            b += Pi @ pos
        gate_pos = np.linalg.lstsq(A, b, rcond=None)[0]

        yaw, yaw_std = self._yaw_from_width_variation(usable, gate_pos)
        return GateEstimate(
            position   = gate_pos,
            yaw        = yaw,
            yaw_std    = yaw_std,
            method     = 'triangulation',
            num_frames = len(usable),
        )

    # ── Strategy 3 — Aspect ratio ────────────────────────────────────────────

    def _estimate_aspect_ratio(
        self, drone_pos: np.ndarray, ray: np.ndarray
    ) -> GateEstimate:
        f_px     = self._K[0, 0]
        bbox_h   = max(self.bbox_height or 1.0, 1.0)
        distance = f_px * self.GATE_REAL_HEIGHT / bbox_h
        gate_pos = drone_pos + distance * ray

        snap     = self._history[-1][2]
        yaw, yaw_std = self._yaw_from_width_variation(
            [(drone_pos, ray, snap)], gate_pos
        )
        return GateEstimate(
            position   = gate_pos,
            yaw        = yaw,
            yaw_std    = yaw_std,
            method     = 'aspect_ratio',
            num_frames = 1,
        )

    # ── Yaw from bbox width variation ────────────────────────────────────────

    def _yaw_from_width_variation(
        self, usable: list, gate_pos: np.ndarray
    ) -> Tuple[float, float]:
        """
        Apparent foreshortening encodes the angle α between camera axis
        and gate normal:
            cos(α) = (w_px / h_px) * (H_real / W_real)
        Gate normal yaw = view_yaw ± (π/2 − α).
        Sign ambiguity resolved by circular mean across frames.
        """
        candidates = []
        for drone_pos, _, snap in usable:
            bw = snap['bbox_width']  if isinstance(snap, dict) else snap.bbox_width
            bh = snap['bbox_height'] if isinstance(snap, dict) else snap.bbox_height
            if bw is None or bh is None or bh < 1.0:
                continue

            cos_alpha = np.clip(
                (bw / bh) * (self.GATE_REAL_HEIGHT / self.GATE_REAL_WIDTH),
                -1.0, 1.0,
            )
            alpha    = np.arccos(cos_alpha)
            to_gate  = gate_pos[:2] - drone_pos[:2]
            view_yaw = np.arctan2(to_gate[1], to_gate[0])
            candidates.append(view_yaw + (np.pi / 2.0 - alpha))
            candidates.append(view_yaw - (np.pi / 2.0 - alpha))

        if not candidates:
            to_gate  = gate_pos[:2] - usable[-1][0][:2]
            return float(np.arctan2(to_gate[1], to_gate[0])), float(np.pi)

        angles   = np.array(candidates)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        mean_yaw = float(np.arctan2(sin_mean, cos_mean))
        R        = np.sqrt(sin_mean**2 + cos_mean**2)
        yaw_std  = float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-9, 1.0))))
        return mean_yaw, yaw_std

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_K(self) -> np.ndarray:
        cam  = self.cam
        f_px = (cam.img_width  / 2.0) / np.tan(cam.fov_x / 2.0)
        f_py = (cam.img_height / 2.0) / np.tan(cam.fov_y / 2.0)
        return np.array([
            [f_px, 0,    cam.img_width  / 2.0],
            [0,    f_py, cam.img_height / 2.0],
            [0,    0,    1.0                  ],
        ], dtype=np.float64)

    def _R_cam_to_world(self, drone_state: dict) -> np.ndarray:
        yaw    = drone_state['yaw']
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [ sy,  0,  cy],
            [-cy,  0,  sy],
            [ 0,  -1,  0 ],
        ], dtype=np.float64)

    def _detection_ray(self, drone_state: dict) -> np.ndarray:
        u, v   = self.center
        f_px   = self._K[0, 0]
        f_py   = self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        ray_cam = np.array([
            (u - cx) / f_px,
            (v - cy) / f_py,
            1.0,
        ])
        ray_cam /= np.linalg.norm(ray_cam)
        return self._R_cam_to_world(drone_state) @ ray_cam
    
class DroneFSM:
    SEARCH_YAW_RATE         = np.deg2rad(10)  # rad/step
    HOVER_Z                 = 1.0 # meters
    TAKEOFF_Z               = 1.0 # meters
    TAKEOFF_Z_TOL           = 0.1 # meters
    MAX_GATE_DRIFT          = 0.5 # meters
    PASSING_TRIGGER_DIST    = 0.5 # meters — horizontal distance to gate center for passage detection
    OVERSHOOT_DIST          = 0.5 # meters — how much to overshoot beyond the gate plane for better passage reliability

    def __init__(self, specs: Optional[CameraSpecs] = None, gate_count: int = 5):
        self.specs = specs if specs is not None else CameraSpecs()
        self.detector = GateDetector(self.specs)
        self.gate_count = gate_count

        self.gate_tracker = GateDetection(self.specs)

        self.state: State = State.TAKEOFF
        self.current_gate_idx: int = 0
        self.gate_position: Optional[Tuple[float, float, float]] = None
        self.search_yaw: float = 0.0
        self.last_setpoint: Optional[Setpoint] = None
        self.last_estimation_pos: Optional[np.ndarray] = None  # XY position at last gate estimate

    def step(self, drone_state, camera_frame) -> Setpoint:
        if self.state == State.TAKEOFF:
            setpoint = self._handle_takeoff(drone_state)

        elif self.state == State.SEARCHING:
            setpoint = self._handle_search(drone_state, camera_frame=camera_frame)

        elif self.state == State.FLY_TO_GATE:
            setpoint = self._handle_fly_to_gate(drone_state, camera_frame=camera_frame)

        elif self.state == State.PASSING_GATE:
            setpoint = self._handle_passing_gate(drone_state)
        
        elif self.state == State.PASSED_GATE:
            setpoint = self._handle_passed_gate(drone_state)
        
        else:
            setpoint = self._hold_position(drone_state)

        self.last_setpoint = setpoint
        return setpoint
    
    @property
    def finished(self) -> bool:
        return False
        return self.current_gate_idx >= self.gate_count
    
    # State handlers
    def _handle_takeoff(self, drone_state) -> Setpoint:
        """
        Climb straight up to TAKEOFF_Z, then transition to SEARCH.
        XY and yaw are frozen at the initial position.
        """
        if abs(drone_state['z_global'] - self.TAKEOFF_Z) < self.TAKEOFF_Z_TOL:
            print(f"[TAKEOFF → SEARCH]  Target altitude {self.TAKEOFF_Z} m reached")
            self._transition(State.SEARCHING)
            self.search_yaw = drone_state['yaw']   # reset scan from current yaw

        return Setpoint(
            x=drone_state['x_global'],
            y=drone_state['y_global'],
            z=self.TAKEOFF_Z,
            yaw=drone_state['yaw'],
        )

    def _handle_search(self, drone_state, camera_frame) -> Setpoint:
        raw = self.detector.detect(camera_frame)
        self.gate_tracker.update_from(raw)
        estimate = self.gate_tracker.estimate(drone_state)

        if estimate is not None:
            self.gate_position = estimate.position
            self.last_estimation_pos = np.array([drone_state['x_global'], drone_state['y_global']])
            print(f"[SEARCH → FLY_TO_GATE]  Gate {self.current_gate_idx} "
                  f"detected at {self.gate_position} with yaw {np.degrees(estimate.yaw):.1f}° "
                  f"(method: {estimate.method}, yaw std: {np.degrees(estimate.yaw_std):.1f}°)")
            self._transition(State.FLY_TO_GATE)
            return self._setpoint_toward_gate(drone_state)
        
        # Rotate in place to search for gate
        self.search_yaw += self.SEARCH_YAW_RATE
        return Setpoint(
            x=drone_state['x_global'],
            y=drone_state['y_global'],
            z=self.HOVER_Z,
            yaw=self.search_yaw
        )
    
    def _handle_fly_to_gate(self, drone_state, camera_frame) -> Setpoint:
        drone_pos = np.array([drone_state['x_global'], drone_state['y_global'], drone_state['z_global']])
        
        if self.gate_position is None:
            return self._hold_position(drone_state)
        
        dist_to_gate = np.linalg.norm(self.gate_position[:2] - drone_pos[:2])

        # ── Transition to PASSING_GATE when close enough ──────────────────
        if dist_to_gate < self.PASSING_TRIGGER_DIST:
            # Compute a setpoint OVERSHOOT_DIST past the gate centre
            direction = self.gate_position[:2] - drone_pos[:2]
            direction /= np.linalg.norm(direction)
            overshoot_xy = self.gate_position[:2] + direction * self.OVERSHOOT_DIST

            self.passing_setpoint = Setpoint(
                x=overshoot_xy[0],
                y=overshoot_xy[1],
                z=self.gate_position[2],
                yaw=np.arctan2(direction[1], direction[0]),
            )
            print(f"[FLY_TO_GATE → PASSING_GATE]  Gate {self.current_gate_idx} "
                f"at {dist_to_gate:.2f} m, overshooting to {overshoot_xy}")
            self._transition(State.PASSING_GATE)
            return self.passing_setpoint
        
        # Re-estimate if drone moved 0.5 m since last estimation
        dist_since_last = np.linalg.norm(drone_state['x_global'] - self.last_estimation_pos[0]) + np.linalg.norm(drone_state['y_global'] - self.last_estimation_pos[1])
        
        if dist_since_last >= 0.5:
            raw = self.detector.detect(camera_frame)
            self.gate_tracker.update_from(raw)
            estimate = self.gate_tracker.estimate(drone_state)

            if estimate is not None:
                drift = np.linalg.norm(estimate.position[:2] - self.gate_position[:2])
                if drift < self.MAX_GATE_DRIFT:
                    self.gate_position = estimate.position
                    self.last_estimation_pos = drone_pos[:2]
                    print(f"[FLY_TO_GATE]  Gate position re-estimated with drift {drift:.2f} m: {self.gate_position}")
                else:
                    print(f"[FLY_TO_GATE]  Re-estimation drift {drift:.2f} m too high, ignoring new estimate")
            
            self.last_estimation_pos = drone_pos[:2]  # still update last estimation position to avoid repeated re-estimation
            print(f"[FLY_TO_GATE]  Gate position updated: {self.gate_position}")

        return self._setpoint_toward_gate(drone_state)
    
    def _handle_passed_gate(self, drone_state) -> Setpoint:
        self.current_gate_idx += 1
        self.gate_position = None
        self.gate_tracker.reset()

        if self.finished:
            print("[PASSED_GATE]  Lap complete!")
            return self._hold_position(drone_state)
        
        print(f"[PASSED_GATE → SEARCH]  Looking for gate {self.current_gate_idx}")
        self.search_yaw = drone_state['yaw']   # reset scan from current yaw
        self._transition(State.SEARCHING)
        return self._hold_position(drone_state)
    
    def _handle_passing_gate(self, drone_state) -> Setpoint:
        """
        Fly to a fixed point on the other side of the gate.
        No detection runs in this state.
        Transition to PASSED_GATE once the drone has crossed the gate plane.
        """
        drone_xy   = np.array([drone_state['x_global'], drone_state['y_global']])
        dist_to_gate = np.linalg.norm(self.gate_position[:2] - drone_xy)

        if dist_to_gate < 0.3:   # drone has crossed the gate plane
            print(f"[PASSING_GATE → PASSED_GATE]  Gate {self.current_gate_idx} passed")
            self._transition(State.PASSED_GATE)
            return self._hold_position(drone_state)

        return self.passing_setpoint   # frozen setpoint on the other side

    
    # Helper methods
    def _transition(self, new_state: State):
        self.state = new_state

    def _hold_position(self, drone_state) -> Setpoint:
        return Setpoint(
            x=drone_state['x_global'],
            y=drone_state['y_global'],
            z=drone_state['z_global'],
            yaw=drone_state['yaw']
        )
    
    def _setpoint_toward_gate(self, drone_state) -> Setpoint:
        gp  = self.gate_position
        
        # yaw computation
        dx  = gp[0] - drone_state['x_global']
        dy  = gp[1] - drone_state['y_global']
        yaw = np.arctan2(dy, dx)

        #print(f"[Setpoint]  Moving toward gate at {gp} with yaw {np.degrees(yaw):.1f}°")
        return Setpoint(
            x=gp[0],
            y=gp[1],
            z=gp[2],
            yaw=yaw,
        )

# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
