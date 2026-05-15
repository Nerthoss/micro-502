import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

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
    TAKEOFF     = auto()
    SEARCHING   = auto()
    ADJUST_Z    = auto()
    GO_TO_GATE  = auto()
    GO_TO_YAW   = auto()
    PASS_GATE   = auto()

@dataclass
class GateDetection:
    detected: bool
    center: Optional[Tuple[int, int]] = None                 # (u, v) pixel coordinates
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None    # (x, y, w, h) bounding box
    world_pos: Optional[Tuple[float, float, float]] = None   # (x, y, z) in world frame
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
    lateral_offset_px: float = 0.0  # tune this to correct systematic left/right bias:
                                     # positive → shift principal point right (gate was estimated too far left)
                                     # negative → shift principal point left  (gate was estimated too far right)

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
              f"z={control_command.z:.2f}  yaw={control_command.yaw:.1f}°")
    
        return list(control_command)
    
class GateDetector:
    HSV_LOWER = np.array([140, 50, 120])
    HSV_UPPER = np.array([160, 255, 255])

    GATE_REAL_HEIGHT = 0.4  # meters

    MIN_CONTOUR_AREA = 200.0  # pixels


    def __init__(self, specs: CameraSpecs):
        self.specs = specs

    def detect(self, frame) -> GateDetection:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)    
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.HSV_LOWER, self.HSV_UPPER)

        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return GateDetection(detected=False)

        best = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        area = w * h

        if area < self.MIN_CONTOUR_AREA:
            return GateDetection(detected=False)

        # Use image moments for the centroid instead of the bounding box centre.
        # The moment centroid is the centre-of-mass of the detected pink pixels,
        # which is robust to partial occlusion: if both sides of the gate frame
        # are visible their pixel masses balance around the true gate centre,
        # whereas the bounding box centre is pulled toward whichever side has
        # more visible pixels.
        M = cv2.moments(best)
        if M['m00'] > 0:
            center_px = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        else:
            center_px = (x + w / 2, y + h / 2)  # fallback to bbox centre

        # Approximate contour to polygon and extract corners if 4 vertices found
        epsilon = 0.02 * cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, epsilon, True)
        corners = tuple(tuple(pt[0]) for pt in approx) if len(approx) == 4 else None
        
        return GateDetection(detected=True, center=center_px, bbox_xywh=(x, y, w, h), corners=corners)
    
    def estimate_world_position(self, detection: GateDetection, drone_state) -> np.ndarray:
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
        # Normalised image-plane coordinates (OpenCV origin = top-left).
        # lateral_offset_px shifts the assumed principal point to compensate
        # for camera mounting offset or calibration bias.
        x_cam = (u - (cam.img_width  / 2.0 + cam.lateral_offset_px)) / f_px   # positive → right
        y_cam = (v -  cam.img_height / 2.0)                           / f_px   # positive → down

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
    
class DroneFSM:
    HOVER_Z                 = 1.0   # meters
    TAKEOFF_Z               = 1.0   # meters
    TAKEOFF_Z_TOL           = 0.05  # meters
    POS_XY_TOL              = 0.05  # meters — tolerance for XY waypoints
    POS_Z_TOL               = 0.05  # meters — tolerance for Z waypoint
    YAW_TOL                 = np.deg2rad(3)  # rad — tolerance for yaw alignment
    SEARCH_LATERAL_OFFSET   =  1.0  # meters (right in drone frame, toward world centre)
    NUM_SEARCH_DETECTIONS   = 5     # number of detections to fuse
    PASS_GATE_FORWARD       = 0.5   # meters to move forward after gate alignment
    EMA_ALPHA               = 0.3   # weight for new detections in GO_TO_GATE (0=ignore, 1=replace)
    APPROACH_OFFSET         = 0.3   # meters in front of gate to stop before passing
    MAX_DETECTION_DRIFT     = 0.5   # meters — max allowed shift per EMA update in GO_TO_GATE

    def __init__(self, specs: Optional[CameraSpecs] = None, gate_count: int = 5):
        self.specs = specs if specs is not None else CameraSpecs()
        self.detector = GateDetector(self.specs)
        self.gate_count = gate_count

        self.state: State = State.TAKEOFF
        self.current_gate_idx: int = 0

        # Gate estimate storage
        self.gate_estimate: Optional[GateEstimate] = None  # fused estimate

        # Search sub-state tracking
        self._search_origin: Optional[np.ndarray] = None   # drone XYZ at start of SEARCHING
        self._search_origin_yaw: float = 0.0
        self._search_detections: list = []                  # accumulates world positions
        self._search_phase: str = 'move_lateral'            # 'move_lateral' | 'detect' | 'return'
        self._search_detect_positions: list = []            # 5 XY positions for detection sweep
        self._search_detect_idx: int = 0                    # which detection position we're at

        self.last_setpoint: Optional[Setpoint] = None

    def step(self, drone_state, camera_frame) -> Setpoint:
        if self.state == State.TAKEOFF:
            setpoint = self._handle_takeoff(drone_state)

        elif self.state == State.SEARCHING:
            setpoint = self._handle_search(drone_state, camera_frame)

        elif self.state == State.ADJUST_Z:
            setpoint = self._handle_adjust_z(drone_state)

        elif self.state == State.GO_TO_GATE:
            setpoint = self._handle_go_to_gate(drone_state, camera_frame)

        elif self.state == State.GO_TO_YAW:
            setpoint = self._handle_go_to_yaw(drone_state)

        elif self.state == State.PASS_GATE:
            setpoint = self._handle_pass_gate(drone_state)

        else:
            setpoint = self._hold_position(drone_state)

        self.last_setpoint = setpoint
        return setpoint

    @property
    def finished(self) -> bool:
        return self.current_gate_idx >= self.gate_count

    # ------------------------------------------------------------------ #
    #  State handlers                                                       #
    # ------------------------------------------------------------------ #

    def _handle_takeoff(self, drone_state) -> Setpoint:
        """Climb to TAKEOFF_Z, then transition to SEARCHING."""
        if abs(drone_state['z_global'] - self.TAKEOFF_Z) < self.TAKEOFF_Z_TOL:
            print(f"[TAKEOFF → SEARCHING]  Altitude {self.TAKEOFF_Z} m reached")
            self._init_search(drone_state)
            self._transition(State.SEARCHING)

        return Setpoint(
            x=drone_state['x_global'],
            y=drone_state['y_global'],
            z=self.TAKEOFF_Z,
            yaw=drone_state['yaw'],
        )

    def _handle_search(self, drone_state, camera_frame) -> Setpoint:
        """
        Three-phase search:
          1. move_lateral  – displace 1 m to the left of world centre
          2. detect        – visit 5 spread positions and collect detections
          3. return        – fly back to the origin of the search
        After phase 3, fuse the collected positions and advance to ADJUST_Z.
        """
        x  = drone_state['x_global']
        y  = drone_state['y_global']
        z  = drone_state['z_global']
        ox, oy, oz = self.search_origin_xyz

        # ── Phase 1: move to lateral offset position ───────────────────────────
        if self._search_phase == 'move_lateral':
            # Offset is in the DRONE frame at search start: -1 m along drone's left axis.
            # Drone forward = [cos(yaw), sin(yaw)]; drone left = [-sin(yaw), cos(yaw)]
            yaw0 = self._search_origin_yaw
            target_x = ox + self.SEARCH_LATERAL_OFFSET * (-np.sin(yaw0))
            target_y = oy + self.SEARCH_LATERAL_OFFSET * ( np.cos(yaw0))
            target_z = self.HOVER_Z
            dist = np.hypot(x - target_x, y - target_y)
            if dist < self.POS_XY_TOL:
                print("[SEARCH]  Lateral offset reached — starting 5-point detection")
                self._search_phase = 'detect'
                self._build_detect_positions(target_x, target_y)
                self._search_detect_idx = 0

            return Setpoint(x=target_x, y=target_y, z=target_z, yaw=self._search_origin_yaw)

        # ── Phase 2: collect detections from 5 positions ──────────────────────
        elif self._search_phase == 'detect':
            tp = self._search_detect_positions[self._search_detect_idx]
            dist = np.hypot(x - tp[0], y - tp[1])

            if dist < self.POS_XY_TOL:
                # We are at the detection position — attempt detection
                detection = self.detector.detect(camera_frame)
                if detection.detected:
                    wp = self.detector.estimate_world_position(detection, drone_state)
                    self._search_detections.append(wp)
                    print(f"[SEARCH]  Detection {len(self._search_detections)}/5 "
                          f"at pos {tp} → gate {wp}")
                else:
                    print(f"[SEARCH]  No detection at pos {tp}")

                self._search_detect_idx += 1
                if self._search_detect_idx >= len(self._search_detect_positions):
                    print("[SEARCH]  5 positions visited — returning to origin")
                    self._search_phase = 'return'

            return Setpoint(x=tp[0], y=tp[1], z=self.HOVER_Z, yaw=self._search_origin_yaw)

        # ── Phase 3: return to origin, then fuse and advance ──────────────────
        elif self._search_phase == 'return':
            dist = np.hypot(x - ox, y - oy)
            if dist < self.POS_XY_TOL:
                self._fuse_detections_and_advance(drone_state)
            return Setpoint(x=ox, y=oy, z=self.HOVER_Z, yaw=self._search_origin_yaw)

        return self._hold_position(drone_state)

    def _handle_adjust_z(self, drone_state) -> Setpoint:
        """Move to the estimated gate Z height; keep XY fixed."""
        gz = self.gate_estimate.position[2]
        if abs(drone_state['z_global'] - gz) < self.POS_Z_TOL:
            print(f"[ADJUST_Z → GO_TO_GATE]  Z {gz:.2f} m reached")
            self._transition(State.GO_TO_GATE)

        return Setpoint(
            x=drone_state['x_global'],
            y=drone_state['y_global'],
            z=gz,
            yaw=drone_state['yaw'],
        )

    def _handle_go_to_gate(self, drone_state, camera_frame) -> Setpoint:
        """
        Fly to the estimated gate XY position at gate Z height.
        Every step, attempt a new detection and refine the estimate
        using an exponential moving average (alpha = EMA_ALPHA).
        """
        # ── Continuous detection: refine gate estimate ──────────────────────
        detection = self.detector.detect(camera_frame)
        if detection.detected:
            new_pos = self.detector.estimate_world_position(detection, drone_state)

            # Reject detections that have jumped too far from the current estimate
            # — this filters out the next gate appearing in the camera view.
            drift = np.linalg.norm(new_pos[:2] - self.gate_estimate.position[:2])
            if drift > self.MAX_DETECTION_DRIFT:
                print(f"[GO_TO_GATE]  Detection rejected — drift {drift:.2f} m "
                      f"> {self.MAX_DETECTION_DRIFT} m (likely next gate)")
            else:
                # EMA update on position
                old_pos = self.gate_estimate.position
                refined_pos = self.EMA_ALPHA * new_pos + (1.0 - self.EMA_ALPHA) * old_pos

                # Gate yaw is its normal direction: bearing-to-gate + π/2
                dx = new_pos[0] - drone_state['x_global']
                dy = new_pos[1] - drone_state['y_global']
                new_yaw  = self._wrap_angle(np.arctan2(dy, dx) + np.pi / 2)
                yaw_diff = self._wrap_angle(new_yaw - self.gate_estimate.yaw)
                refined_yaw = self._wrap_angle(
                    self.gate_estimate.yaw + self.EMA_ALPHA * yaw_diff
                )

                self.gate_estimate = GateEstimate(
                    position=refined_pos,
                    yaw=refined_yaw,
                    yaw_std=self.gate_estimate.yaw_std,
                    method='ema',
                    num_frames=self.gate_estimate.num_frames + 1,
                )
                print(f"[GO_TO_GATE]  Estimate refined (n={self.gate_estimate.num_frames}): "
                      f"pos={np.round(refined_pos, 2)}, yaw={np.degrees(refined_yaw):.1f}°")

        # ── Navigation: target is APPROACH_OFFSET metres in front of the gate ──
        # Gate normal direction = gate_estimate.yaw.
        # "In front" = offset opposite to the normal (drone approaches from that side).
        gx, gy, gz = self.gate_estimate.position
        gate_yaw = self.gate_estimate.yaw
        approach_x = gx - self.APPROACH_OFFSET * np.cos(gate_yaw)
        approach_y = gy - self.APPROACH_OFFSET * np.sin(gate_yaw)

        dx = approach_x - drone_state['x_global']
        dy = approach_y - drone_state['y_global']
        dist = np.hypot(dx, dy)

        if dist < self.POS_XY_TOL:
            print("[GO_TO_GATE → GO_TO_YAW]  Approach point reached")
            self._transition(State.GO_TO_YAW)

        # Face toward the gate (not the approach point) while flying
        dx_gate = gx - drone_state['x_global']
        dy_gate = gy - drone_state['y_global']
        yaw = np.arctan2(dy_gate, dx_gate) if np.hypot(dx_gate, dy_gate) > 0.01 else drone_state['yaw']
        return Setpoint(x=approach_x, y=approach_y, z=gz, yaw=yaw)

    def _handle_go_to_yaw(self, drone_state) -> Setpoint:
        """Rotate in yaw to match the gate's estimated normal direction."""
        target_yaw = self.gate_estimate.yaw
        yaw_err = self._wrap_angle(target_yaw - drone_state['yaw'])

        if abs(yaw_err) < self.YAW_TOL:
            print(f"[GO_TO_YAW → PASS_GATE]  Yaw aligned to "
                  f"{np.degrees(target_yaw):.1f}°")
            self._transition(State.PASS_GATE)

        gx, gy, gz = self.gate_estimate.position
        return Setpoint(x=gx, y=gy, z=gz, yaw=target_yaw)

    def _handle_pass_gate(self, drone_state) -> Setpoint:
        """
        Move forward (along current yaw) by PASS_GATE_FORWARD metres,
        then start searching for the next gate.
        """
        if not hasattr(self, '_pass_gate_target') or self._pass_gate_target is None:
            yaw = drone_state['yaw']
            self._pass_gate_target = np.array([
                drone_state['x_global'] + self.PASS_GATE_FORWARD * np.cos(yaw),
                drone_state['y_global'] + self.PASS_GATE_FORWARD * np.sin(yaw),
                drone_state['z_global'],
            ])
            print(f"[PASS_GATE]  Moving forward {self.PASS_GATE_FORWARD} m "
                  f"to {self._pass_gate_target[:2]}")

        tx, ty, tz = self._pass_gate_target
        dist = np.hypot(drone_state['x_global'] - tx, drone_state['y_global'] - ty)

        if dist < self.POS_XY_TOL:
            self._pass_gate_target = None
            self.current_gate_idx += 1
            self.gate_estimate = None

            if self.finished:
                print("[PASS_GATE]  All gates passed — holding position.")
                return self._hold_position(drone_state)

            print(f"[PASS_GATE → SEARCHING]  Gate {self.current_gate_idx - 1} complete, "
                  f"searching for gate {self.current_gate_idx}")
            self._init_search(drone_state)
            self._transition(State.SEARCHING)
            return self._hold_position(drone_state)

        return Setpoint(x=tx, y=ty, z=tz, yaw=drone_state['yaw'])

    # ------------------------------------------------------------------ #
    #  Search helpers                                                        #
    # ------------------------------------------------------------------ #

    def _init_search(self, drone_state):
        """Reset all search sub-state to start a fresh search cycle."""
        self._search_origin = np.array([
            drone_state['x_global'],
            drone_state['y_global'],
            drone_state['z_global'],
        ])
        self._search_origin_yaw = drone_state['yaw']
        self._search_detections = []
        self._search_phase = 'move_lateral'
        self._search_detect_positions = []
        self._search_detect_idx = 0

    @property
    def search_origin_xyz(self):
        if self._search_origin is None:
            return 0.0, 0.0, self.HOVER_Z
        return self._search_origin[0], self._search_origin[1], self._search_origin[2]

    def _build_detect_positions(self, base_x: float, base_y: float):
        """
        Build 5 slightly spread positions around the lateral offset point
        so each detection comes from a different viewpoint.
        Offsets of ±0.2 m along the drone's forward axis (at search start)
        to get parallax without losing sight of the gate.
        """
        offsets = [-0.2, -0.1, 0.0, 0.1, 0.2]
        yaw0 = self._search_origin_yaw
        # Drone forward unit vector in world frame
        fwd_x = np.cos(yaw0)
        fwd_y = np.sin(yaw0)
        self._search_detect_positions = [
            (base_x + d * fwd_x, base_y + d * fwd_y) for d in offsets
        ]

    def _fuse_detections_and_advance(self, drone_state):
        """Average valid detections into a gate estimate, then go to ADJUST_Z.
        If no detections were collected, rotate 10° CCW and restart the manoeuvre."""
        if not self._search_detections:
            print("[SEARCH]  No detections — rotating 10° CCW and restarting")
            rotated_yaw = self._wrap_angle(self._search_origin_yaw + np.deg2rad(10))
            self._init_search(drone_state)
            self._search_origin_yaw = rotated_yaw  # override with rotated yaw
            return

        positions = np.array(self._search_detections)   # (N, 3)
        mean_pos  = positions.mean(axis=0)

        # Gate yaw is the gate's normal direction (perpendicular to its face).
        # arctan2 gives the bearing from drone to gate; the gate normal is
        # 90° counter-clockwise from that bearing (i.e. bearing + π/2).
        dx = mean_pos[0] - drone_state['x_global']
        dy = mean_pos[1] - drone_state['y_global']
        gate_yaw = self._wrap_angle(np.arctan2(dy, dx) + np.pi / 2)

        self.gate_estimate = GateEstimate(
            position=mean_pos,
            yaw=gate_yaw,
            yaw_std=0.0,
            method='fusion',
            num_frames=len(self._search_detections),
        )
        print(f"[SEARCH → ADJUST_Z]  Gate estimate fused from "
              f"{len(self._search_detections)} detections: pos={mean_pos}, "
              f"yaw={np.degrees(gate_yaw):.1f}°")
        self._transition(State.ADJUST_Z)

    # ------------------------------------------------------------------ #
    #  Generic helpers                                                       #
    # ------------------------------------------------------------------ #

    def _transition(self, new_state: State):
        self.state = new_state

    def _hold_position(self, drone_state) -> Setpoint:
        return Setpoint(
            x=drone_state['x_global'],
            y=drone_state['y_global'],
            z=drone_state['z_global'],
            yaw=drone_state['yaw'],
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
