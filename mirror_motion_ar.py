"""
Mirror Motion AR
Requirements: pip install opencv-python mediapipe trimesh pyrender
Run: python mirqror_motion_ar.py
Keys: 1-4=color  +/-=size  S=skeleton  Space=capture  Q=quit
"""

import cv2
import mediapipe as mp
import numpy as np
import sys, os, datetime

try:
    import trimesh as _trimesh
    import pyrender as _pyrender
    _3D_OK = True
except ImportError:
    _3D_OK = False
    print("trimesh/pyrender not found — 3D models disabled. pip install trimesh pyrender")

# ===== CONFIG =====
LIMB_TYPE        = "right_arm_below"
MODEL_SCALE      = 1.0
LIMB_COLOR_IDX   = 1
ACTIVE_MODEL_IDX = None   # None = procedural; int = index into _models list
CAM_W, CAM_H     = 640, 480
UI_W             = 182

# ===== ANTHROPOMETRIC RATIOS =====
# Expected (missing) limb length as fraction of body height
# Source: standard anatomical proportions
ANTHRO_RATIOS = {
    'arm_below': 0.254,   # forearm + hand  (elbow → fingertip)
    'arm_above': 0.440,   # full arm        (shoulder → fingertip)
    'leg_below': 0.285,   # shin + foot     (knee → floor)
    'leg_above': 0.530,   # full leg        (hip → floor)
}

# Best-fit initial MODEL_SCALE per limb type
# Derived from: expected_limb / (ref_segment × 2)
OPTIMAL_SCALE = {
    'right_arm_below': 0.68,   # (forearm+hand) / (upper_arm×2)
    'left_arm_below':  0.68,
    'right_arm_above': 1.00,   # full arm — ref already covers it
    'left_arm_above':  1.00,
    'right_leg_below': 0.58,   # (shin+foot) / (thigh×2)
    'left_leg_below':  0.58,
    'right_leg_above': 1.00,   # full leg — ref already covers it
    'left_leg_above':  1.00,
}

# Color schemes (BGR)
LIMB_COLORS = {
    1: {'base': (96,  148, 198), 'light': (130, 180, 225), 'dark': (60,  108, 155), 'name': 'Skin'},
    2: {'base': (155, 185,  35), 'light': (185, 215,  75), 'dark': (105, 130,  10), 'name': 'Cyan'},
    3: {'base': (190,  50, 170), 'light': (215,  90, 200), 'dark': (135,  15, 120), 'name': 'Purple'},
    4: {'base': (155, 158, 165), 'light': (195, 198, 205), 'dark': (108, 110, 118), 'name': 'Gray'},
}

# ===== MEDIAPIPE POSE =====
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        MODEL_PATH)

options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_detector = mp_vision.PoseLandmarker.create_from_options(options)

# ===== LANDMARK HELPERS =====
LM = {
    'r_shoulder': 12, 'l_shoulder': 11,
    'r_elbow': 14,    'l_elbow': 13,
    'r_wrist': 16,    'l_wrist': 15,
    'r_hip': 24,      'l_hip': 23,
    'r_knee': 26,     'l_knee': 25,
    'r_ankle': 28,    'l_ankle': 27,
}

# Exponential moving average smoothing
_smooth_state = {}
SMOOTH_FACTOR = 0.35

def get_lm(landmarks, key, w, h, min_vis=0.3):
    idx = LM.get(key)
    if idx is None or idx >= len(landmarks):
        return None
    p = landmarks[idx]
    if p.visibility < min_vis:
        return None
    raw = np.array([p.x * w, p.y * h])
    if key in _smooth_state:
        _smooth_state[key] = SMOOTH_FACTOR * raw + (1 - SMOOTH_FACTOR) * _smooth_state[key]
    else:
        _smooth_state[key] = raw.copy()
    return _smooth_state[key].copy()

def dist2d(a, b):
    return np.linalg.norm(b - a)

import time as _time
_leg_debug_last = [0.0]
def _leg_debug(mode, hip, other, ref_len):
    now = _time.time()
    if now - _leg_debug_last[0] > 2.0:
        _leg_debug_last[0] = now
        other_str = f"({other[0]:.0f},{other[1]:.0f})" if other is not None else "None"
        print(f"[leg_above] mode={mode}  hip=({hip[0]:.0f},{hip[1]:.0f})  "
              f"other={other_str}  ref_len={ref_len:.1f}")

def get_attach_info(landmarks, limb_type, w, h):
    side       = 'r' if 'right' in limb_type else 'l'
    other_side = 'l' if side == 'r' else 'r'
    is_below   = 'below' in limb_type
    is_arm     = 'arm'   in limb_type

    if is_arm:
        shoulder = get_lm(landmarks, side + '_shoulder', w, h)
        elbow    = get_lm(landmarks, side + '_elbow',    w, h)
        wrist    = get_lm(landmarks, side + '_wrist',    w, h)

        if is_below:
            # Below elbow: attach at elbow, continue upper-arm direction
            if elbow is None:
                return None
            direction = (elbow - shoulder) if shoulder is not None else np.array([0., 1.])
            ref_len = dist2d(shoulder, elbow) if shoulder is not None else h * 0.15
            return (elbow.copy(), direction, ref_len)
        else:
            # Below shoulder: attach at shoulder, full arm
            if shoulder is None:
                return None
            if wrist is not None:
                direction = wrist - shoulder
                ref_len   = dist2d(shoulder, wrist) / 2.0
            elif elbow is not None:
                direction = elbow - shoulder
                ref_len   = dist2d(shoulder, elbow)
            else:
                hip      = get_lm(landmarks, side + '_hip', w, h)
                other_sh = get_lm(landmarks, other_side + '_shoulder', w, h)
                if hip is not None:
                    direction = hip - shoulder
                elif other_sh is not None:
                    sv = other_sh - shoulder
                    direction = np.array([-sv[1], sv[0]])
                    if direction[1] < 0:
                        direction = -direction
                else:
                    direction = np.array([0., 1.])
                ref_len = h * 0.22
            return (shoulder.copy(), direction, ref_len)
    else:
        # Lower min_vis for legs — MediaPipe scores them lower than upper body
        hip   = get_lm(landmarks, side + '_hip',   w, h, min_vis=0.1)
        knee  = get_lm(landmarks, side + '_knee',  w, h, min_vis=0.1)
        ankle = get_lm(landmarks, side + '_ankle', w, h, min_vis=0.1)
        if is_below:
            if knee is None:
                return None
            direction = (ankle - knee) if ankle is not None else \
                        (knee - hip if hip is not None else np.array([0., 1.]))
            ref_len = dist2d(hip, knee) if hip is not None else h * 0.20
            return (knee.copy(), direction, ref_len)
        else:
            # Full leg: attach at hip
            if hip is None:
                return None
            if ankle is not None:
                direction = ankle - hip
                ref_len   = dist2d(hip, ankle) / 2.0
                _leg_debug('ankle', hip, ankle, ref_len)
            elif knee is not None:
                direction = knee - hip
                ref_len   = dist2d(hip, knee)
                _leg_debug('knee', hip, knee, ref_len)
            else:
                direction = np.array([0., 1.])
                ref_len   = h * 0.22
                _leg_debug('fallback', hip, None, ref_len)
            return (hip.copy(), direction, ref_len)

# ===== LIMB DRAWING =====

def _seg(img, p1, p2, diam, colors):
    """
    Draw a smooth cylindrical segment using layered antialiased lines.
    Each layer is narrower and lighter, producing a 3D gradient effect.
    """
    a = (int(p1[0]), int(p1[1]))
    b = (int(p2[0]), int(p2[1]))
    dk = colors['dark']
    bs = colors['base']
    li = colors['light']
    md = tuple(int((dk[i] + bs[i]) // 2) for i in range(3))
    ml = tuple(int((bs[i] + li[i]) // 2) for i in range(3))

    # From widest/darkest (shadow) to narrowest/lightest (highlight)
    for frac, color in ((1.15, dk), (1.00, md), (0.84, bs), (0.56, ml), (0.30, li)):
        t = max(1, int(diam * frac))
        cv2.line(img, a, b, color, t, cv2.LINE_AA)


def _dot(img, pt, r, colors):
    """Shaded circle (joint / cap)."""
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), max(1, int(r * 1.15)), colors['dark'],  -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), max(1, int(r)),         colors['base'],  -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), max(1, int(r * 0.45)),  colors['light'], -1, cv2.LINE_AA)


def draw_arm(img, attach_pt, direction, arm_len, limb_type, colors):
    """Realistic arm drawn with antialiased gradient lines.
    arm_above: full arm (shoulder→upper arm→elbow→forearm→hand)
    arm_below: below elbow only (elbow→forearm→hand)
    """
    is_right = 'right' in limb_type
    is_above = 'above' in limb_type
    perp = np.array([-direction[1], direction[0]])

    if is_above:
        # Upper arm: 42% of total arm_len, wider at shoulder
        upper_frac = 0.42
        elbow_pt = attach_pt + direction * arm_len * upper_frac

        for i in range(3):
            t1 = i / 3
            t2 = (i + 1) / 3
            p1 = attach_pt + direction * (arm_len * upper_frac * t1)
            p2 = attach_pt + direction * (arm_len * upper_frac * t2)
            r = arm_len * (0.092 - t1 * 0.020)
            _seg(img, p1, p2, r * 2, colors)

        _dot(img, elbow_pt, arm_len * 0.062, colors)

        fore_attach = elbow_pt
        fore_len    = arm_len * 0.58
    else:
        fore_attach = attach_pt
        fore_len    = arm_len

    # --- Forearm: 5 tapered segments (wider at elbow, narrower at wrist) ---
    wrist = fore_attach + direction * fore_len * 0.63
    for i in range(5):
        t1 = i / 5
        t2 = (i + 1) / 5
        p1 = fore_attach + direction * (fore_len * 0.63 * t1)
        p2 = fore_attach + direction * (fore_len * 0.63 * t2)
        r = fore_len * (0.082 - t1 * 0.026)
        _seg(img, p1, p2, r * 2, colors)

    _dot(img, wrist, fore_len * 0.056, colors)

    # --- Palm ---
    knuckle = wrist + direction * fore_len * 0.185
    _seg(img, wrist, knuckle, fore_len * 0.19, colors)

    # --- Four fingers with slight natural fan ---
    fan = [(-0.068, 0.997, 0.115),
           (-0.020, 0.999, 0.150),
           ( 0.025, 0.999, 0.140),
           ( 0.068, 0.996, 0.098)]
    fw = fore_len * 0.040

    for perp_off, dir_w, f_len_rel in fan:
        fb = knuckle + perp * fore_len * perp_off
        fd = (direction * dir_w + perp * perp_off * 0.15)
        fd = fd / np.linalg.norm(fd)
        ft = fb + fd * fore_len * f_len_rel
        _seg(img, fb, ft, fw, colors)
        _dot(img, ft, fw * 0.52, colors)

    # --- Thumb ---
    ts = -1 if is_right else 1
    tb = fore_attach + direction * fore_len * 0.44 + perp * fore_len * 0.074 * ts
    td = direction * 0.40 + perp * ts * 0.916
    td /= np.linalg.norm(td)
    tt = tb + td * fore_len * 0.108
    _seg(img, tb, tt, fore_len * 0.060, colors)
    _dot(img, tt, fore_len * 0.030, colors)

    # Attachment dot
    cv2.circle(img, (int(attach_pt[0]), int(attach_pt[1])), 7, (0, 230, 140), -1, cv2.LINE_AA)
    cv2.circle(img, (int(attach_pt[0]), int(attach_pt[1])), 7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_leg(img, attach_pt, direction, leg_len, limb_type, colors):
    """Realistic leg drawn with antialiased gradient lines.
    leg_above: full leg (hip→thigh→knee→shin→foot)
    leg_below: below knee only (knee→shin→foot)
    """
    is_right = 'right' in limb_type
    is_above = 'above' in limb_type
    perp = np.array([-direction[1], direction[0]])

    if is_above:
        # Full leg: thigh (46%) + knee joint + shin (54%)
        thigh_frac = 0.46
        knee_pt = attach_pt + direction * leg_len * thigh_frac

        n_thigh = 4
        for i in range(n_thigh):
            t1 = i / n_thigh
            t2 = (i + 1) / n_thigh
            p1 = attach_pt + direction * (leg_len * thigh_frac * t1)
            p2 = attach_pt + direction * (leg_len * thigh_frac * t2)
            taper = 1.0 - t1 * 0.30
            r = leg_len * 0.090 * taper
            _seg(img, p1, p2, r * 2, colors)

        _dot(img, knee_pt, leg_len * 0.072, colors)

        shin_start = knee_pt
        shin_len   = leg_len * 0.54
    else:
        shin_start = attach_pt
        shin_len   = leg_len

    # --- Shin: 5 tapered segments with calf curve ---
    ankle = shin_start + direction * shin_len * 0.75
    n_shin = 5
    calf_peak = 0.35
    for i in range(n_shin):
        t1 = i / n_shin
        t2 = (i + 1) / n_shin
        p1 = shin_start + direction * (shin_len * 0.75 * t1)
        p2 = shin_start + direction * (shin_len * 0.75 * t2)
        calf_bump = max(0.0, 1 - ((t1 - calf_peak) / 0.40) ** 2)
        r = shin_len * (0.072 + calf_bump * 0.032)
        _seg(img, p1, p2, r * 2, colors)

    _dot(img, ankle, shin_len * 0.062, colors)

    # --- Foot ---
    foot_side = 1 if is_right else -1
    foot_raw = perp * foot_side
    horiz = np.array([float(foot_side), 0.0])
    foot_dir = 0.55 * foot_raw + 0.45 * horiz
    fdn = np.linalg.norm(foot_dir)
    if fdn < 1e-6:
        return
    foot_dir /= fdn

    toe  = ankle + foot_dir * shin_len * 0.26
    heel = ankle - foot_dir * shin_len * 0.09

    _seg(img, heel, toe, shin_len * 0.100, colors)
    _dot(img, toe,  shin_len * 0.058, colors)
    _dot(img, heel, shin_len * 0.068, colors)

    # Attachment dot
    cv2.circle(img, (int(attach_pt[0]), int(attach_pt[1])), 7, (0, 230, 140), -1, cv2.LINE_AA)
    cv2.circle(img, (int(attach_pt[0]), int(attach_pt[1])), 7, (255, 255, 255), 2, cv2.LINE_AA)


def render_limb(frame, attach_info, scale, limb_type, color_idx):
    attach_pt, direction, ref_len = attach_info
    dir_len = np.linalg.norm(direction)
    if dir_len < 1e-6:
        direction = np.array([0., 1.])
    else:
        direction = direction / dir_len

    limb_len = ref_len * scale * 2.0
    colors   = LIMB_COLORS[color_idx]
    overlay  = frame.copy()

    if 'arm' in limb_type:
        draw_arm(overlay, attach_pt, direction, limb_len, limb_type, colors)
    else:
        draw_leg(overlay, attach_pt, direction, limb_len, limb_type, colors)

    cv2.addWeighted(overlay, 0.93, frame, 0.07, 0, frame)
    return frame

# ===== SKELETON =====
def draw_skeleton(frame, landmarks, w, h):
    connections = [
        ('r_shoulder','l_shoulder'), ('r_shoulder','r_elbow'), ('r_elbow','r_wrist'),
        ('l_shoulder','l_elbow'),    ('l_elbow','l_wrist'),
        ('r_shoulder','r_hip'),      ('l_shoulder','l_hip'),    ('r_hip','l_hip'),
        ('r_hip','r_knee'),          ('r_knee','r_ankle'),
        ('l_hip','l_knee'),          ('l_knee','l_ankle'),
    ]
    for a_key, b_key in connections:
        a = get_lm(landmarks, a_key, w, h, min_vis=0.2)
        b = get_lm(landmarks, b_key, w, h, min_vis=0.2)
        if a is not None and b is not None:
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (0, 200, 180), 1)
    for key in LM:
        p = get_lm(landmarks, key, w, h, min_vis=0.2)
        if p is not None:
            cv2.circle(frame, tuple(p.astype(int)), 3, (0, 220, 180), -1)

# ===== 3D PROCEDURAL LIMB =====
_SPRITE_SIZE = 400

# Four skin styles: (display name, RGB color)
_SKIN_STYLES = [
    ('Natural',  [255, 160, 122]),   # 0xffa07a — light salmon skin
    ('Bionic',   [180, 185, 200]),   # metallic silver
    ('Neon',     [ 80, 200, 255]),   # electric blue
    ('Dark',     [100,  75,  60]),   # dark tone
]



def _concat(*parts):
    return _trimesh.util.concatenate(list(parts))

_R90X = _trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

def _cyl(radius, height, sections=20):
    """Cylinder aligned with Y axis (trimesh default is Z)."""
    m = _trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_transform(_R90X)
    return m


# ── Geometry builders (limb pointing in +Y, attach point at y=0) ──────────────

def _geo_arm_below():
    """Forearm + wrist + palm  (elbow → fingertip)."""
    forearm = _cyl(0.15, 1.2)
    forearm.apply_translation([0, -0.6, 0])

    wrist = _trimesh.creation.icosphere(radius=0.13, subdivisions=2)
    wrist.apply_translation([0, -1.2, 0])

    # Palm: wider, flatter box
    palm = _trimesh.creation.box(extents=[0.38, 0.32, 0.09])
    palm.apply_translation([0, -1.46, 0])

    # Four finger stubs
    finger_xs = [-0.12, -0.04, 0.04, 0.12]
    fingers = []
    for fx in finger_xs:
        f = _cyl(0.03, 0.22)
        f.apply_translation([fx, -1.73, 0])
        fingers.append(f)

    # Thumb stub
    thumb = _cyl(0.035, 0.18)
    thumb.apply_transform(_trimesh.transformations.rotation_matrix(np.pi/5, [0, 0, 1]))
    thumb.apply_translation([0.20, -1.38, 0])

    return _concat(forearm, wrist, palm, *fingers, thumb)


def _geo_arm_above():
    """Full arm  (shoulder → fingertip)."""
    upper = _cyl(0.20, 1.40)
    upper.apply_translation([0, -0.70, 0])

    elbow = _trimesh.creation.icosphere(radius=0.18, subdivisions=2)
    elbow.apply_translation([0, -1.40, 0])

    forearm = _cyl(0.14, 1.10)
    forearm.apply_translation([0, -1.95, 0])

    wrist = _trimesh.creation.icosphere(radius=0.12, subdivisions=2)
    wrist.apply_translation([0, -2.50, 0])

    hand = _trimesh.creation.box(extents=[0.30, 0.28, 0.10])
    hand.apply_translation([0, -2.74, 0])

    return _concat(upper, elbow, forearm, wrist, hand)


def _geo_leg_below():
    """Shin + ankle + foot  (knee → floor)."""
    shin = _cyl(0.18, 1.50)
    shin.apply_translation([0, -0.75, 0])

    ankle = _trimesh.creation.icosphere(radius=0.15, subdivisions=2)
    ankle.apply_translation([0, -1.50, 0])

    foot = _trimesh.creation.box(extents=[0.20, 0.55, 0.14])
    foot.apply_translation([0.12, -1.78, 0])

    return _concat(shin, ankle, foot)


def _geo_leg_above():
    """Full leg  (hip → floor)."""
    thigh = _cyl(0.26, 1.80)
    thigh.apply_translation([0, -0.90, 0])

    knee = _trimesh.creation.icosphere(radius=0.22, subdivisions=2)
    knee.apply_translation([0, -1.80, 0])

    shin = _cyl(0.18, 1.40)
    shin.apply_translation([0, -2.50, 0])

    ankle = _trimesh.creation.icosphere(radius=0.15, subdivisions=2)
    ankle.apply_translation([0, -3.20, 0])

    foot = _trimesh.creation.box(extents=[0.20, 0.55, 0.14])
    foot.apply_translation([0.12, -3.48, 0])

    return _concat(thigh, knee, shin, ankle, foot)


_GEO_BUILDERS = {
    'arm_below': _geo_arm_below,
    'arm_above': _geo_arm_above,
    'leg_below': _geo_leg_below,
    'leg_above': _geo_leg_above,
}


def _bake_sprite(mesh, size=_SPRITE_SIZE, rgb=None):
    """Render mesh orthographically (camera along -Z). Returns BGRA ndarray."""
    if rgb is not None:
        mat = _pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[rgb[0]/255, rgb[1]/255, rgb[2]/255, 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.75,
        )
        try:
            py_mesh = _pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
        except Exception:
            py_mesh = _pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False)
    else:
        try:
            py_mesh = _pyrender.Mesh.from_trimesh(mesh, smooth=True)
        except Exception:
            py_mesh = _pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = _pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(py_mesh)

    cam = _pyrender.OrthographicCamera(xmag=1.6, ymag=2.2)
    cp = np.eye(4); cp[2, 3] = 5.0
    scene.add(cam, pose=cp)

    light = _pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5)
    lp = np.eye(4); lp[2, 3] = 5.0
    scene.add(light, pose=lp)

    renderer = _pyrender.OffscreenRenderer(size, size)
    color, depth = renderer.render(scene)
    renderer.delete()

    alpha = (depth > 0).astype(np.uint8) * 255
    bgr   = color[:, :, ::-1]
    return np.dstack([bgr, alpha])


class _SkinModel:
    """Procedural 3D limb — no file loading, always available."""
    def __init__(self, name, rgb):
        self.name    = name
        self.rgb     = rgb
        self.ok      = True
        self._cache  = {}   # limb_key → BGRA sprite

    def get_sprite(self, limb_type):
        key = ('arm_below' if 'arm' in limb_type and 'below' in limb_type else
               'arm_above' if 'arm' in limb_type else
               'leg_below' if 'below' in limb_type else 'leg_above')
        if key not in self._cache:
            mesh = _GEO_BUILDERS[key]()
            self._cache[key] = _bake_sprite(mesh, rgb=self.rgb)
        return self._cache[key]


def _composite_model(frame, model, attach_info, scale, limb_type):
    """Rotate & scale the model sprite to fit the limb, then alpha-blend onto frame."""
    attach_pt, direction, ref_len = attach_info
    dir_len = np.linalg.norm(direction)
    direction = direction / dir_len if dir_len > 1e-6 else np.array([0., 1.])

    limb_len_px = ref_len * scale * 2.0
    mid_pt = attach_pt + direction * (limb_len_px / 2.0)

    sprite = model.get_sprite(limb_type)
    alpha_ch = sprite[:, :, 3]
    rows = np.where(np.any(alpha_ch > 10, axis=1))[0]
    cols = np.where(np.any(alpha_ch > 10, axis=0))[0]
    if len(rows) == 0:
        return

    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    cropped = sprite[rmin:rmax + 1, cmin:cmax + 1]
    ch, cw = cropped.shape[:2]
    if ch == 0:
        return

    scale_f = limb_len_px / ch
    new_h = max(4, int(ch * scale_f))
    new_w = max(4, int(cw * scale_f))
    scaled = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dir_angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    rot_deg   = dir_angle - 90.0

    cx, cy = new_w / 2.0, new_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    ca, sa = abs(M[0, 0]), abs(M[0, 1])
    out_w = int(new_h * sa + new_w * ca) + 4
    out_h = int(new_h * ca + new_w * sa) + 4
    M[0, 2] += (out_w - new_w) / 2.0
    M[1, 2] += (out_h - new_h) / 2.0

    rotated = cv2.warpAffine(scaled, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

    x0 = int(mid_pt[0]) - out_w // 2
    y0 = int(mid_pt[1]) - out_h // 2

    fh, fw = frame.shape[:2]
    rh, rw = rotated.shape[:2]
    sx0 = max(0, -x0);      sy0 = max(0, -y0)
    ex  = min(rw, fw - x0); ey  = min(rh, fh - y0)
    if ex <= sx0 or ey <= sy0:
        return
    fx0 = x0 + sx0; fy0 = y0 + sy0
    r_crop = rotated[sy0:ey, sx0:ex]
    f_crop = frame[fy0:fy0 + (ey - sy0), fx0:fx0 + (ex - sx0)]
    a = r_crop[:, :, 3:4].astype(float) / 255.0
    frame[fy0:fy0 + (ey - sy0), fx0:fx0 + (ex - sx0)] = (
        a * r_crop[:, :, :3] + (1 - a) * f_crop
    ).astype(np.uint8)


_models = [_SkinModel(n, c) for n, c in _SKIN_STYLES] if _3D_OK else []

# ===== UI PANEL =====
LIMB_NAMES = {
    'right_arm_below': 'Right Arm - Below Elbow',
    'right_arm_above': 'Right Arm - Below Shoulder',
    'left_arm_below':  'Left Arm  - Below Elbow',
    'left_arm_above':  'Left Arm  - Below Shoulder',
    'right_leg_below': 'Right Leg - Below Knee',
    'right_leg_above': 'Right Leg - Full Leg',
    'left_leg_below':  'Left Leg  - Below Knee',
    'left_leg_above':  'Left Leg  - Full Leg',
}

_BTN_DEFS = [
    ('right_arm_below', 'Right Arm',  'Below Elbow'),
    ('right_arm_above', 'Right Arm',  'Below Shoulder'),
    ('left_arm_below',  'Left Arm',   'Below Elbow'),
    ('left_arm_above',  'Left Arm',   'Below Shoulder'),
    ('right_leg_below', 'Right Leg',  'Below Knee'),
    ('right_leg_above', 'Right Leg',  'Full Leg'),
    ('left_leg_below',  'Left Leg',   'Below Knee'),
    ('left_leg_above',  'Left Leg',   'Full Leg'),
    ('__gap__',         '',           ''),
    ('size_minus',      '-',          ''),
    ('size_plus',       '+',          ''),
    ('skeleton',        'Skeleton',   ''),
    ('__gap__',         '',           ''),
    ('__label__',       '3D SKINS',   ''),
    ('model_0',         'Natural',    'Skin'),
    ('model_1',         'Bionic',     'Silver'),
    ('model_2',         'Neon',       'Blue'),
    ('model_3',         'Dark',       'Tone'),
    ('model_none',      'Procedural', '2D'),
    ('__gap__',         '',           ''),
    ('screenshot',      'Capture',    'Space'),
]

def _build_buttons(panel_x):
    btns = []
    x1 = panel_x + 8
    x2 = panel_x + UI_W - 8
    bw = x2 - x1
    bh = 28
    gap = 4
    y = 38

    i = 0
    while i < len(_BTN_DEFS):
        bid, l1, l2 = _BTN_DEFS[i]
        if bid == '__gap__':
            y += 10
            i += 1
            continue
        if bid == '__label__':
            btns.append({'id': '__label__', 'line1': l1, 'line2': '',
                         'x1': x1, 'y1': y, 'x2': x2, 'y2': y + 16})
            y += 20
            i += 1
            continue
        if bid == 'size_minus':
            mid = x1 + bw // 2 - 3
            btns.append({'id': 'size_minus', 'line1': ' - ', 'line2': '',
                         'x1': x1, 'y1': y, 'x2': mid, 'y2': y + bh})
            btns.append({'id': 'size_plus',  'line1': ' + ', 'line2': '',
                         'x1': mid + 6, 'y1': y, 'x2': x2, 'y2': y + bh})
            y += bh + gap
            i += 2
            continue
        btns.append({'id': bid, 'line1': l1, 'line2': l2,
                     'x1': x1, 'y1': y, 'x2': x2, 'y2': y + bh})
        y += bh + gap
        i += 1
    return btns


def draw_panel(canvas, btns, limb_type, show_skeleton, model_scale, color_idx):
    h = canvas.shape[0]
    px = CAM_W

    canvas[:, px:] = (22, 22, 22)
    cv2.line(canvas, (px, 0), (px, h), (70, 70, 70), 1)

    # Title
    cv2.putText(canvas, "SELECT LIMB", (px + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 150), 1)

    for btn in btns:
        bid = btn['id']

        # Section label — draw as plain text, no button box
        if bid == '__label__':
            cv2.putText(canvas, btn['line1'],
                        (btn['x1'], btn['y1'] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 200, 160), 1)
            continue

        model_active_id = (f"model_{ACTIVE_MODEL_IDX}"
                           if ACTIVE_MODEL_IDX is not None else 'model_none')
        is_active = (bid == limb_type) or \
                    (bid == 'skeleton' and show_skeleton) or \
                    (bid.startswith('model_') and bid == model_active_id)

        # Dim model buttons that failed to load
        unavailable = False
        if bid.startswith('model_') and bid != 'model_none':
            idx = int(bid.split('_')[1])
            unavailable = _models and not _models[idx].ok

        bg = (0, 120, 60) if is_active else (30, 30, 30) if unavailable else (50, 50, 50)
        fg = (255, 255, 255) if is_active else (80, 80, 80) if unavailable else (205, 205, 205)

        cv2.rectangle(canvas, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']), bg, -1)
        cv2.rectangle(canvas, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']), (80, 80, 80), 1)

        bh_btn = btn['y2'] - btn['y1']
        bx = btn['x1'] + 5
        if btn['line2']:
            cv2.putText(canvas, btn['line1'], (bx, btn['y1'] + bh_btn // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, fg, 1)
            cv2.putText(canvas, btn['line2'], (bx, btn['y1'] + bh_btn // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31, fg, 1)
        else:
            cv2.putText(canvas, btn['line1'], (bx, btn['y1'] + bh_btn // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, fg, 1)

    # Color swatches
    last_y = max(b['y2'] for b in btns) if btns else 360
    swatch_y = last_y + 14
    cv2.putText(canvas, "Color  1-4:", (px + 8, swatch_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    swatch_y += 10
    sw_size = 22
    sw_gap  = 6
    sw_x = px + 8
    for idx, col in LIMB_COLORS.items():
        x0, y0 = sw_x + (idx - 1) * (sw_size + sw_gap), swatch_y
        cv2.rectangle(canvas, (x0, y0), (x0 + sw_size, y0 + sw_size), col['base'], -1)
        border = (255, 255, 255) if idx == color_idx else (80, 80, 80)
        thickness = 2 if idx == color_idx else 1
        cv2.rectangle(canvas, (x0, y0), (x0 + sw_size, y0 + sw_size), border, thickness)
        cv2.putText(canvas, str(idx), (x0 + 7, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    # Scale + quit hint
    info_y = swatch_y + sw_size + 14
    cv2.putText(canvas, f"Scale: {model_scale:.1f}",
                (px + 8, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    cv2.putText(canvas, "Q = quit",
                (px + 8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)


# ===== ONBOARDING FORM =====

def show_onboarding():
    """Show tkinter setup form. Returns user-profile dict or None if cancelled."""
    import tkinter as tk

    # ── Palette ──────────────────────────────
    BG     = '#12121e'
    CARD   = '#1c1c30'
    FIELD  = '#252540'
    GREEN  = '#00c896'
    RED    = '#e94560'
    FG     = '#eaeaea'
    FG2    = '#8888aa'

    result = [None]

    root = tk.Tk()
    root.title("Mirror Motion AR — Setup")
    root.configure(bg=BG)
    root.resizable(False, True)
    W = 460
    sx, sy = root.winfo_screenwidth(), root.winfo_screenheight()
    win_h = min(560, sy - 80)
    root.geometry(f"{W}x{win_h}+{(sx - W)//2}+{(sy - win_h)//2}")
    root.minsize(W, 400)

    # ── Helpers ─────────────────────────────
    def lbl(parent, text, size=10, bold=False, color=FG, bg=None):
        return tk.Label(parent, text=text, bg=bg or parent['bg'], fg=color,
                        font=('Segoe UI', size, 'bold' if bold else 'normal'))

    def mini_field(parent, var, w=10):
        f = tk.Frame(parent, bg=FIELD, padx=1, pady=1)
        tk.Entry(f, textvariable=var, bg=FIELD, fg=FG,
                 insertbackground=FG, relief='flat',
                 font=('Segoe UI', 10), bd=4, width=w).pack()
        return f

    def radio(parent, text, var, val, bg=None):
        b = bg or parent['bg']
        return tk.Radiobutton(parent, text=text, variable=var, value=val,
                              bg=b, fg=FG, selectcolor=FIELD,
                              activebackground=b, activeforeground=GREEN,
                              font=('Segoe UI', 10))

    def sep_grid(parent, row, pad=4):
        tk.Frame(parent, bg='#2a2a4a', height=1).grid(
            row=row, column=0, columnspan=4, sticky='ew', pady=pad)

    # ── Header (fixed) ───────────────────────
    hdr = tk.Frame(root, bg='#0a0a18', pady=8)
    hdr.pack(fill='x', side='top')
    lbl(hdr, "Mirror Motion AR", 16, bold=True, color=GREEN).pack()
    lbl(hdr, "Patient Setup", 9, color=FG2).pack(pady=(1, 0))

    # ── Footer + START button (fixed at bottom) ──
    footer = tk.Frame(root, bg='#0a0a1a', pady=10)
    footer.pack(fill='x', side='bottom')

    err_lbl = lbl(footer, "", 9, color=RED, bg='#0a0a1a')
    err_lbl.pack(pady=(0, 4))

    # ── Scrollable body ──────────────────────
    scroll_container = tk.Frame(root, bg=BG)
    scroll_container.pack(fill='both', expand=True, side='top')

    vscroll = tk.Scrollbar(scroll_container, orient='vertical', bg=BG)
    vscroll.pack(side='right', fill='y')

    canvas_inner = tk.Canvas(scroll_container, bg=BG, highlightthickness=0,
                             yscrollcommand=vscroll.set)
    canvas_inner.pack(side='left', fill='both', expand=True)
    vscroll.config(command=canvas_inner.yview)

    body = tk.Frame(canvas_inner, bg=BG, padx=24, pady=8)
    body_window = canvas_inner.create_window((0, 0), window=body, anchor='nw')

    def _on_body_configure(event):
        canvas_inner.configure(scrollregion=canvas_inner.bbox('all'))
        canvas_inner.itemconfig(body_window, width=canvas_inner.winfo_width())
    body.bind('<Configure>', _on_body_configure)
    canvas_inner.bind('<Configure>',
                      lambda e: canvas_inner.itemconfig(body_window, width=e.width))

    def _on_mousewheel(event):
        canvas_inner.yview_scroll(int(-1 * (event.delta / 120)), 'units')
    root.bind_all('<MouseWheel>', _on_mousewheel)

    for c in (1, 3):
        body.columnconfigure(c, weight=1)

    def pair(r, l1, v1, l2, v2):
        lbl(body, l1, 9, color=FG2).grid(row=r, column=0, sticky='w', pady=2)
        mini_field(body, v1).grid(row=r, column=1, sticky='ew', padx=(8,14), pady=2)
        lbl(body, l2, 9, color=FG2).grid(row=r, column=2, sticky='w', pady=2)
        mini_field(body, v2).grid(row=r, column=3, sticky='ew', padx=(8,0), pady=2)

    name_var   = tk.StringVar()
    height_var = tk.StringVar(value='170')
    weight_var = tk.StringVar(value='70')
    age_var    = tk.StringVar()

    pair(0, "Name (optional)", name_var, "Age (optional)", age_var)
    pair(1, "Height  (cm) *",  height_var, "Weight (kg) *",  weight_var)

    sep_grid(body, row=2)

    # ── Amputation section ───────────────────
    lbl(body, "Amputation Details", 11, bold=True).grid(
        row=3, column=0, columnspan=4, sticky='w', pady=(4, 2))

    side_frame = tk.Frame(body, bg=BG)
    side_frame.grid(row=4, column=0, columnspan=4, sticky='w')
    lbl(side_frame, "Side:", 10, color=FG2).pack(side='left', padx=(0, 10))
    side_var = tk.StringVar(value='right')
    radio(side_frame, "Right", side_var, 'right').pack(side='left', padx=6)
    radio(side_frame, "Left",  side_var, 'left').pack(side='left', padx=6)

    lbl(body, "Amputation Level:", 10, color=FG2).grid(
        row=5, column=0, columnspan=4, sticky='w', pady=(6, 2))

    lvl_frame = tk.Frame(body, bg=BG)
    lvl_frame.grid(row=6, column=0, columnspan=4, sticky='w')
    level_var = tk.StringVar(value='arm_below')

    arm_box = tk.Frame(lvl_frame, bg=CARD, padx=10, pady=6)
    arm_box.pack(side='left', padx=(0, 12))
    lbl(arm_box, "ARM", 9, bold=True, color=FG2, bg=CARD).pack(anchor='w')
    radio(arm_box, "Below elbow",    level_var, 'arm_below', CARD).pack(anchor='w', pady=1)
    radio(arm_box, "Below shoulder", level_var, 'arm_above', CARD).pack(anchor='w', pady=1)

    leg_box = tk.Frame(lvl_frame, bg=CARD, padx=10, pady=6)
    leg_box.pack(side='left')
    lbl(leg_box, "LEG", 9, bold=True, color=FG2, bg=CARD).pack(anchor='w')
    radio(leg_box, "Below knee", level_var, 'leg_below', CARD).pack(anchor='w', pady=1)
    radio(leg_box, "Full leg",   level_var, 'leg_above', CARD).pack(anchor='w', pady=1)

    sep_grid(body, row=7)

    # ── Result box ───────────────────────────
    res_box = tk.Frame(body, bg=CARD, padx=16, pady=8)
    res_box.grid(row=8, column=0, columnspan=4, sticky='ew', pady=2)
    lbl(res_box, "Calculated limb length:", 9, color=FG2, bg=CARD).pack(anchor='w')
    res_lbl = lbl(res_box, "—", 20, bold=True, color=GREEN, bg=CARD)
    res_lbl.pack(anchor='w')
    res_sub = lbl(res_box, "", 9, color=FG2, bg=CARD)
    res_sub.pack(anchor='w')

    # ── Size slider ──────────────────────────
    size_row = tk.Frame(body, bg=BG)
    size_row.grid(row=9, column=0, columnspan=4, sticky='ew', pady=(6, 2))
    lbl(size_row, "Initial size:", 9, color=FG2).pack(side='left', padx=(0, 10))
    scale_var = tk.DoubleVar(value=1.0)
    size_val_lbl = lbl(size_row, "1.0×", 10, bold=True, color=GREEN)
    size_val_lbl.pack(side='right', padx=(8, 0))
    slider = tk.Scale(size_row, variable=scale_var, from_=0.5, to=2.0,
                      resolution=0.1, orient='horizontal',
                      bg=BG, fg=FG, highlightthickness=0,
                      troughcolor=FIELD, activebackground=GREEN,
                      showvalue=False, length=200)
    slider.pack(side='left', fill='x', expand=True)
    def on_scale_move(*_):
        size_val_lbl.config(text=f"{scale_var.get():.1f}×")
    scale_var.trace_add('write', on_scale_move)

    # ── Dynamic update ───────────────────────
    def update(*_):
        try:
            h   = float(height_var.get())
            lvl = level_var.get()
            cm  = ANTHRO_RATIOS[lvl] * h
            res_lbl.config(text=f"{cm:.1f} cm")
            descs = {
                'arm_below': 'forearm + hand  (elbow → fingertip)',
                'arm_above': 'full arm  (shoulder → fingertip)',
                'leg_below': 'shin + foot  (knee → floor)',
                'leg_above': 'full leg  (hip → floor)',
            }
            res_sub.config(text=descs.get(lvl, ''))
        except Exception:
            res_lbl.config(text="—")
            res_sub.config(text="")

    height_var.trace_add('write', update)
    level_var.trace_add('write',  update)
    update()

    def on_start():
        err_lbl.config(text="")
        try:
            h    = float(height_var.get())
            w_kg = float(weight_var.get())
            if not (100 <= h  <= 250): raise ValueError("Height must be 100–250 cm")
            if not (20  <= w_kg <= 300): raise ValueError("Weight must be 20–300 kg")
        except ValueError as e:
            err_lbl.config(text=str(e))
            return

        side    = side_var.get()
        lvl     = level_var.get()
        lt      = f"{side}_{lvl}"
        limb_cm = round(ANTHRO_RATIOS[lvl] * h, 1)
        base_scale = OPTIMAL_SCALE.get(lt, 0.70)

        result[0] = {
            'name':        name_var.get().strip() or 'User',
            'height_cm':   h,
            'weight_kg':   w_kg,
            'age':         age_var.get().strip(),
            'limb_type':   lt,
            'limb_cm':     limb_cm,
            'model_scale': round(base_scale * scale_var.get(), 2),
        }
        root.destroy()

    start_btn = tk.Button(
        footer, text="▶  START AR SESSION",
        command=on_start,
        bg=GREEN, fg='#050510',
        font=('Segoe UI', 14, 'bold'),
        relief='flat', bd=0,
        padx=0, pady=14,
        cursor='hand2',
        activebackground='#00e6aa',
        activeforeground='#050510',
    )
    start_btn.pack(fill='x', padx=24)

    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.mainloop()
    return result[0]


# ===== MAIN =====
_clicked_btn = [None]

def main():
    global LIMB_TYPE, MODEL_SCALE, LIMB_COLOR_IDX, ACTIVE_MODEL_IDX, _models

    # ── Onboarding form ──────────────────────
    profile = show_onboarding()
    if profile is None:
        print("Setup cancelled.")
        return

    LIMB_TYPE   = profile['limb_type']
    MODEL_SCALE = profile['model_scale']
    user_name   = profile['name']
    expected_cm = profile['limb_cm']

    print(f"Welcome, {user_name}!")
    print(f"Limb: {LIMB_NAMES.get(LIMB_TYPE, LIMB_TYPE)}")
    print(f"Expected limb length: {expected_cm} cm  |  Initial scale: {MODEL_SCALE:.2f}\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    print("Warming up camera...")
    for _ in range(5):
        cap.read()
    print("Ready!  Stand back so your full body is visible.")
    print("Keys: 1-4=color  +/-=size  S=skeleton  Space=capture  Q=quit\n")

    TOTAL_W = CAM_W + UI_W
    btns = _build_buttons(CAM_W)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in btns:
                if btn['x1'] <= x <= btn['x2'] and btn['y1'] <= y <= btn['y2']:
                    _clicked_btn[0] = btn['id']
                    break

    cv2.namedWindow("Mirror Motion AR")
    cv2.setMouseCallback("Mirror Motion AR", mouse_cb)

    show_skeleton  = False
    frame_ts       = 0
    screenshot_msg = None
    save_dir       = os.path.dirname(os.path.abspath(__file__))

    def do_screenshot(canvas):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"capture_{ts}.png")
        cv2.imwrite(path, canvas)
        print(f"Saved: capture_{ts}.png")
        return f"Saved: capture_{ts}.png"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        results  = pose_detector.detect_for_video(mp_image, frame_ts)

        detected = False
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            raw = results.pose_landmarks[0]
            class _LM:
                def __init__(self, x, y, z, v): self.x=x; self.y=y; self.z=z; self.visibility=v
            lms = [_LM(p.x, p.y, p.z, p.visibility) for p in raw]

            if show_skeleton:
                draw_skeleton(frame, lms, w, h)

            attach_info = get_attach_info(lms, LIMB_TYPE, w, h)
            if attach_info is not None:
                use_model = (ACTIVE_MODEL_IDX is not None and
                             _models and _models[ACTIVE_MODEL_IDX].ok)
                if use_model:
                    _composite_model(frame, _models[ACTIVE_MODEL_IDX],
                                     attach_info, MODEL_SCALE, LIMB_TYPE)
                else:
                    frame = render_limb(frame, attach_info, MODEL_SCALE,
                                        LIMB_TYPE, LIMB_COLOR_IDX)

            detected = True

        # ---- Top bar ----
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.putText(frame, "Mirror Motion AR", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(frame, f"{user_name}  |  Expected: {expected_cm} cm",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 150), 1)
        status_text  = "AR active" if detected else "Stand back — show shoulders"
        status_color = (0, 220, 80) if detected else (0, 160, 255)
        dot_x = w - 140
        cv2.circle(frame, (dot_x, 22), 7, status_color, -1)
        cv2.putText(frame, status_text, (dot_x + 14, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1)

        # ---- Bottom bar ----
        bar_h = 42
        overlay_bar = frame.copy()
        cv2.rectangle(overlay_bar, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bar, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, LIMB_NAMES.get(LIMB_TYPE, LIMB_TYPE),
                    (10, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        if ACTIVE_MODEL_IDX is not None and _models:
            model_name = _models[ACTIVE_MODEL_IDX].name
            cv2.putText(frame, f"3D: {model_name}   Scale: {MODEL_SCALE:.1f}",
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 220, 160), 1)
        else:
            color_name = LIMB_COLORS[LIMB_COLOR_IDX]['name']
            cv2.putText(frame, f"Color: {color_name}   Scale: {MODEL_SCALE:.1f}",
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

        # ---- Compose canvas ----
        canvas = np.zeros((CAM_H, TOTAL_W, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        draw_panel(canvas, btns, LIMB_TYPE, show_skeleton, MODEL_SCALE, LIMB_COLOR_IDX)

        # ---- Screenshot notification ----
        if screenshot_msg is not None:
            msg_text, expiry = screenshot_msg
            if datetime.datetime.now().timestamp() < expiry:
                cv2.rectangle(canvas, (8, h // 2 - 24), (w - 8, h // 2 + 24), (0, 0, 0), -1)
                cv2.putText(canvas, msg_text, (16, h // 2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 130), 2)
            else:
                screenshot_msg = None

        cv2.imshow("Mirror Motion AR", canvas)

        # ---- Mouse clicks ----
        clicked = _clicked_btn[0]
        _clicked_btn[0] = None
        if clicked:
            if clicked in LIMB_NAMES:
                LIMB_TYPE = clicked
                print(f"Limb: {LIMB_NAMES[clicked]}")
            elif clicked == 'size_plus':
                MODEL_SCALE = min(MODEL_SCALE + 0.1, 4.0)
            elif clicked == 'size_minus':
                MODEL_SCALE = max(MODEL_SCALE - 0.1, 0.1)
            elif clicked == 'skeleton':
                show_skeleton = not show_skeleton
            elif clicked == 'model_none':
                ACTIVE_MODEL_IDX = None
                print("Mode: Procedural")
            elif clicked.startswith('model_'):
                idx = int(clicked.split('_')[1])
                if _models and _models[idx].ok:
                    ACTIVE_MODEL_IDX = idx
                    print(f"3D model: {_models[idx].name}")
                else:
                    print(f"Model not loaded: {_MODEL_DEFS[idx][1]}")
            elif clicked == 'screenshot':
                msg = do_screenshot(canvas)
                screenshot_msg = (msg, datetime.datetime.now().timestamp() + 2.5)

        # ---- Keyboard ----
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            MODEL_SCALE = min(MODEL_SCALE + 0.1, 4.0)
        elif key == ord('-'):
            MODEL_SCALE = max(MODEL_SCALE - 0.1, 0.1)
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord(' '):
            msg = do_screenshot(canvas)
            screenshot_msg = (msg, datetime.datetime.now().timestamp() + 2.5)
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            LIMB_COLOR_IDX = key - ord('0')
            print(f"Color: {LIMB_COLORS[LIMB_COLOR_IDX]['name']}")
        # Legacy shortcuts
        elif key == ord('a'): LIMB_TYPE = 'right_arm_below'
        elif key == ord('b'): LIMB_TYPE = 'left_arm_below'
        elif key == ord('c'): LIMB_TYPE = 'right_arm_above'
        elif key == ord('d'): LIMB_TYPE = 'left_arm_above'
        elif key == ord('r'): LIMB_TYPE = 'right_leg_below'
        elif key == ord('e'): LIMB_TYPE = 'right_leg_above'
        elif key == ord('l'): LIMB_TYPE = 'left_leg_below'
        elif key == ord('f'): LIMB_TYPE = 'left_leg_above'

    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    print("Closed.")

if __name__ == '__main__':
    main()
