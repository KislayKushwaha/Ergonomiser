import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import time

# =====================================================
# Utility: Angle Calculation
# =====================================================
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# =====================================================
# Kinovea-Style Angle Annotation
# =====================================================
def draw_angle_annotation(img, point_a, vertex, point_c, angle_value, label="", color=(0,255,255)):
    cv2.line(img, vertex, point_a, (80,80,80), 1, cv2.LINE_AA)
    cv2.line(img, vertex, point_c, (80,80,80), 1, cv2.LINE_AA)
    vec_a = np.array(point_a) - np.array(vertex)
    vec_c = np.array(point_c) - np.array(vertex)
    radius = int(min(np.linalg.norm(vec_a), np.linalg.norm(vec_c)) * 0.15)
    radius = max(20, min(radius, 60))
    start = int(np.degrees(np.arctan2(vec_a[1], vec_a[0])))
    end = int(np.degrees(np.arctan2(vec_c[1], vec_c[0])))
    if abs(end - start) > 180:
        if end > start:
            end -= 360
        else:
            start -= 360
    cv2.ellipse(img, vertex, (radius, radius), 0, start, end, color, 2)
    mid = (start + end) / 2
    lx = int(vertex[0] + radius * 1.5 * np.cos(np.radians(mid)))
    ly = int(vertex[1] + radius * 1.5 * np.sin(np.radians(mid)))
    text = f"{label}: {int(angle_value)}" if label else f"{int(angle_value)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (lx-4, ly-th-4), (lx+tw+4, ly+4), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.circle(img, vertex, 4, color, -1)

# =====================================================
# MediaPipe Setup
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# Neck View State (Sticky)
# =====================================================
neck_view_state = {'mode': 'front'}

# =====================================================
# Data Storage
# =====================================================
history = []

def choose_side(landmarks):
    left = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP
    ]
    right = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    lv = np.mean([landmarks[i].visibility for i in left])
    rv = np.mean([landmarks[i].visibility for i in right])
    return 'left' if lv >= rv else 'right'

# =====================================================
# Frame Processing
# =====================================================
def process(frame, mirror=False):
    cam = cv2.flip(frame, 1) if mirror else frame.copy()
    h, w, _ = cam.shape
    res = pose.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
    out = cam.copy()
    if not res.pose_landmarks:
        return out
    lm = res.pose_landmarks.landmark
    P = lambda i: (int(lm[i].x * w), int(lm[i].y * h))
    sides = {}
    for s in ('left', 'right'):
        p = 'LEFT' if s == 'left' else 'RIGHT'
        sides[s] = {
            'shoulder': P(getattr(mp_pose.PoseLandmark, p+'_SHOULDER')),
            'elbow': P(getattr(mp_pose.PoseLandmark, p+'_ELBOW')),
            'wrist': P(getattr(mp_pose.PoseLandmark, p+'_WRIST')),
            'index': P(getattr(mp_pose.PoseLandmark, p+'_INDEX')),
            'hip': P(getattr(mp_pose.PoseLandmark, p+'_HIP')),
            'knee': P(getattr(mp_pose.PoseLandmark, p+'_KNEE')),
            'ankle': P(getattr(mp_pose.PoseLandmark, p+'_ANKLE')),
            'ear': P(getattr(mp_pose.PoseLandmark, p+'_EAR'))
        }
    angles = {}
    for s, p in sides.items():
        angles[s] = {
            'elbow': angle(p['shoulder'], p['elbow'], p['wrist']),
            'wrist': angle(p['elbow'], p['wrist'], p['index']),
            'shoulder': angle(p['elbow'], p['shoulder'], p['hip']),
            'knee': angle(p['hip'], p['knee'], p['ankle']),
            'neck': angle(p['ear'], p['shoulder'], p['hip'])
        }
    
    ls, rs = sides['left']['shoulder'], sides['right']['shoulder']
    lh, rh = sides['left']['hip'], sides['right']['hip']
    shoulder_mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hip_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
    vertical_ref = (hip_mid[0], hip_mid[1] + 100)
    trunk_angle = angle(shoulder_mid, hip_mid, vertical_ref)
    chosen = choose_side(lm)
    
    def is_side_visible(side):
        prefix = 'LEFT' if side == 'left' else 'RIGHT'
        shoulder = lm[getattr(mp_pose.PoseLandmark, prefix + '_SHOULDER')]
        nose = lm[mp_pose.PoseLandmark.NOSE]
        horizontal_dist = abs(shoulder.x - nose.x) * w
        return horizontal_dist > w * 0.08 and shoulder.visibility > 0.4
    
    left_visible = is_side_visible('left')
    right_visible = is_side_visible('right')
    
    if not left_visible and not right_visible:
        if chosen == 'left':
            left_visible = True
        else:
            right_visible = True
    
    mp_draw.draw_landmarks(
        out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(100,100,100), thickness=1, circle_radius=2),
        mp_draw.DrawingSpec(color=(150,150,150), thickness=2)
    )
    
    nose_pt = P(mp_pose.PoseLandmark.NOSE)
    left_ear = sides['left']['ear']
    right_ear = sides['right']['ear']
    dl = np.linalg.norm(np.array(nose_pt) - np.array(left_ear))
    dr = np.linalg.norm(np.array(nose_pt) - np.array(right_ear))
    ear_ratio = min(dl, dr) / max(dl, dr)
    
    FRONT_ENTER = 0.78
    SIDE_ENTER = 0.65
    if neck_view_state['mode'] == 'front':
        if ear_ratio < SIDE_ENTER:
            neck_view_state['mode'] = 'side'
    else:
        if ear_ratio > FRONT_ENTER:
            neck_view_state['mode'] = 'front'
    
    is_front_view = (neck_view_state['mode'] == 'front')
    farthest_side = 'left' if dl > dr else 'right'
    farthest_ear = left_ear if farthest_side == 'left' else right_ear
    
    if is_front_view:
        draw_angle_annotation(out, nose_pt, shoulder_mid, hip_mid, 
                              angle(nose_pt, shoulder_mid, hip_mid), "Neck", (255,200,100))
    else:
        if farthest_side == 'left' and left_visible:
            draw_angle_annotation(out, farthest_ear, shoulder_mid, hip_mid, 
                                  angle(farthest_ear, shoulder_mid, hip_mid), "L-Neck", (255,200,100))
        if farthest_side == 'right' and right_visible:
            draw_angle_annotation(out, farthest_ear, shoulder_mid, hip_mid, 
                                  angle(farthest_ear, shoulder_mid, hip_mid), "R-Neck", (255,200,100))
    
    for side in ('left', 'right'):
        if side == 'left' and not left_visible:
            continue
        if side == 'right' and not right_visible:
            continue
        pts = sides[side]
        ang = angles[side]
        lbl = 'L' if side == 'left' else 'R'
        draw_angle_annotation(out, pts['elbow'], pts['shoulder'], pts['hip'], 
                              ang['shoulder'], f"{lbl}-Shoulder", (100,255,200))
        draw_angle_annotation(out, pts['shoulder'], pts['elbow'], pts['wrist'], 
                              ang['elbow'], f"{lbl}-Elbow", (200,100,255))
        draw_angle_annotation(out, pts['elbow'], pts['wrist'], pts['index'], 
                              ang['wrist'], f"{lbl}-Wrist", (255,255,100))
        draw_angle_annotation(out, pts['hip'], pts['knee'], pts['ankle'], 
                              ang['knee'], f"{lbl}-Knee", (150,255,150))
    
    draw_angle_annotation(out, shoulder_mid, hip_mid, vertical_ref, trunk_angle, "Trunk", (255,150,100))
    
    history.append({
        'time': time.time(),
        'trunk': trunk_angle,
        'left_wrist': angles['left']['wrist'],
        'right_wrist': angles['right']['wrist']
    })
    return out

# =====================================================
# INPUT SELECTION
# =====================================================
print("1 - Webcam")
print("2 - Video File")
print("3 - Image File")
mode = input("Select mode (1/2/3): ").strip()

cap = None
image = None
mirror = False
writer = None

if mode == "1":
    cap = cv2.VideoCapture(0)
    mirror = True
elif mode == "2":
    path = input("Enter video path: ").strip().strip('"')
    cap = cv2.VideoCapture(path)
    mirror_input = input("Is this a webcam/mirror recording? (y/n): ").strip().lower()
    mirror = mirror_input == 'y'

    Path("outputs").mkdir(exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        "outputs/annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
elif mode == "3":
    image = cv2.imread(input("Enter image path: ").strip().strip('"'))
else:
    exit("Invalid mode")

# =====================================================
# MAIN LOOP
# =====================================================
if cap:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process(frame, mirror)
        cv2.imshow("Ergonomiser", processed)
        if writer:
            writer.write(processed)
        if cv2.waitKey(10) & 0xFF in (ord('q'), ord('Q')):
            break
    cap.release()
    if writer:
        writer.release()
else:
    processed = process(image)
    cv2.imshow("Ergonomiser", processed)
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/annotated_image.png", processed)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# =====================================================
# EXPORT
# =====================================================
if history:
    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv("outputs/joint_angles_report.csv", index=False)
