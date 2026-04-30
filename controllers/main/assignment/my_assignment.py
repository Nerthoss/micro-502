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
    FLY_TO_GATE = auto()
    PASSED_GATE = auto()

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
        
        center_px = (x + w / 2, y + h / 2)

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
    
class DroneFSM:
    SEARCH_YAW_RATE         = np.deg2rad(10)  # rad/step
    HOVER_Z                 = 1.0 # meters
    TAKEOFF_Z               = 1.0 # meters
    TAKEOFF_Z_TOL           = 0.1 # meters
    MAX_GATE_DRIFT          = 0.5 # meters

    def __init__(self, specs: Optional[CameraSpecs] = None, gate_count: int = 5):
        self.specs = specs if specs is not None else CameraSpecs()
        self.detector = GateDetector(self.specs)
        self.gate_count = gate_count

        self.state: State = State.TAKEOFF
        self.current_gate_idx: int = 0
        self.gate_position: Optional[Tuple[float, float, float]] = None
        self.search_yaw: float = 0.0
        self.last_setpoint: Optional[Setpoint] = None
        self.last_estimation_pos: Optional[np.ndarray] = None  # XY position at last gate estimate

    def step(self, drone_state, camera_frame) -> Setpoint:
        detection = (
            self.detector.detect(camera_frame)
            if self.state == State.SEARCHING
            else GateDetection(detected=False)
        )

        if self.state == State.TAKEOFF:
            setpoint = self._handle_takeoff(drone_state)

        elif self.state == State.SEARCHING:
            setpoint = self._handle_search(drone_state, detection)

        elif self.state == State.FLY_TO_GATE:
            setpoint = self._handle_fly_to_gate(drone_state, camera_frame)
        
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

    def _handle_search(self, drone_state, detection: GateDetection) -> Setpoint:
        if detection.detected:
            self.gate_position = self.detector.estimate_world_position(detection, drone_state)
            self.last_estimation_pos = np.array([drone_state['x_global'], drone_state['y_global']])
            print(f"[SEARCH → FLY_TO_GATE]  Gate {self.current_gate_idx} "
                  f"detected at {self.gate_position}")
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
        
        # Re-estimate if drone moved 0.5 m since last estimation
        dist_since_last = np.linalg.norm(drone_state['x_global'] - self.last_estimation_pos[0]) + np.linalg.norm(drone_state['y_global'] - self.last_estimation_pos[1])
        if dist_since_last >= 0.5:
            detection = self.detector.detect(camera_frame)
            if detection.detected:
                new_est = self.detector.estimate_world_position(detection, drone_state)
                drift = np.linalg.norm(new_est[:2] - self.gate_position[:2])
                if drift < self.MAX_GATE_DRIFT:
                    self.gate_position = new_est
                    self.last_estimation_pos = drone_pos[:2]
                    print(f"[FLY_TO_GATE]  Gate position re-estimated with drift {drift:.2f} m: {self.gate_position}")
                else:
                    print(f"[FLY_TO_GATE]  Re-estimation drift {drift:.2f} m too high, ignoring new estimate")
                    self.last_estimation_pos = drone_pos[:2]  # still update last estimation position to avoid repeated re-estimation
            print(f"[FLY_TO_GATE]  Gate position updated: {self.gate_position}")

        # Passage detection: drone has reached the gate plane
        dist = np.linalg.norm(self.gate_position[:2] - drone_pos[:2])  # horizontal distance to gate center
        if dist < 0.1:  # metres — tune this threshold
            print(f"[FLY_TO_GATE → PASSED_GATE]  Gate {self.current_gate_idx} passed")
            self._transition(State.PASSED_GATE)
            return self._hold_position(drone_state)

        return self._setpoint_toward_gate(drone_state)
    
    def _handle_passed_gate(self, drone_state) -> Setpoint:
        self.current_gate_idx += 1
        self.gate_position = None

        if self.finished:
            print("[PASSED_GATE]  Lap complete!")
            return self._hold_position(drone_state)
        
        print(f"[PASSED_GATE → SEARCH]  Looking for gate {self.current_gate_idx}")
        self.search_yaw = drone_state['yaw']   # reset scan from current yaw
        self._transition(State.SEARCHING)
        return self._hold_position(drone_state)
    
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
