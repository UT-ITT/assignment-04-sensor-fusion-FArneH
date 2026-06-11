import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import pyglet.shapes
import random
import sys
from PIL import Image

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

cap = cv2.VideoCapture(video_id)
ret, frame0 = cap.read()
if not ret:
    print("kamera geht nicht")
    sys.exit(1)

CAM_H, CAM_W = frame0.shape[:2]

# fenster groesse berechnen damit es auf den bildschirm passt
screen = pyglet.display.get_display().get_default_screen()
scale = min((screen.width * 0.9) / CAM_W, (screen.height * 0.85) / CAM_H, 1.5)
WIN_W = int(CAM_W * scale)
WIN_H = int(CAM_H * scale)

window = pyglet.window.Window(WIN_W, WIN_H, "AR Bubble Catcher")

# aruco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
params = aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 53
params.adaptiveThreshWinSizeStep = 4
params.minMarkerPerimeterRate = 0.02
params.errorCorrectionRate = 1.0
detector = aruco.ArucoDetector(aruco_dict, params)

# spielvariablen
score = 0
missed = 0
bubbles = []
pop_effects = []
spawn_timer = 0.0
SPAWN_INTERVAL = 1.8
MAX_BUBBLES = 4
BUBBLE_RADIUS = 40

# tracking variablen
current_img = None
finger_x = None
finger_y = None
last_warp = None
smooth_x = None
smooth_y = None


def cv2glet(img):
    rows, cols, ch = img.shape
    raw = Image.fromarray(img).tobytes()
    return pyglet.image.ImageData(cols, rows, 'RGB', raw, pitch=-cols * ch)


def cam_to_win(x, y):
    wx = int(x / CAM_W * window.width)
    wy = int((CAM_H - y) / CAM_H * window.height)
    return wx, wy


def get_warp_matrix(corners):
    centers = np.array([c[0].mean(axis=0) for c in corners])
    s = centers.sum(axis=1)
    d = np.diff(centers, axis=1).flatten()
    src = np.array([
        centers[np.argmin(s)],
        centers[np.argmin(d)],
        centers[np.argmax(s)],
        centers[np.argmax(d)],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [CAM_W-1, 0], [CAM_W-1, CAM_H-1], [0, CAM_H-1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def find_finger(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hautfarbe erkennen
    mask = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([20, 200, 255]))

    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 4000:
        return None, None

    mom = cv2.moments(biggest)
    if mom['m00'] == 0:
        return None, None

    cx = mom['m10'] / mom['m00']
    cy = mom['m01'] / mom['m00']

    # fingerspitze suchen: randpunkte überspringen weil der arm immer
    # am bildrand reinkommt, die fingerspitze ist weiter innen
    hull = cv2.convexHull(biggest)
    tip_x, tip_y = int(cx), int(cy)
    max_dist = 0
    margin = 40
    for pt in hull:
        px, py = int(pt[0][0]), int(pt[0][1])
        if px < margin or px > CAM_W - margin or py < margin or py > CAM_H - margin:
            continue
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d > max_dist:
            max_dist = d
            tip_x, tip_y = px, py

    return tip_x, tip_y


def spawn_bubble():
    if sum(1 for b in bubbles if b['alive']) >= MAX_BUBBLES:
        return
    y = float(random.randint(BUBBLE_RADIUS + 40, int(CAM_H * 0.75)))
    speed = random.uniform(90, 160)
    bubbles.append({'x': float(CAM_W - BUBBLE_RADIUS), 'y': y, 'vx': -speed, 'alive': True})


spawn_bubble()


def update(dt):
    global current_img, finger_x, finger_y, spawn_timer, smooth_x, smooth_y, last_warp, score, missed

    ret, frame = cap.read()
    if not ret:
        return

    # board erkennen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) >= 4:
        last_warp = get_warp_matrix(corners[:4])

    if last_warp is not None:
        warped = cv2.warpPerspective(frame, last_warp, (CAM_W, CAM_H))
    else:
        warped = frame

    # finger tracken
    rx, ry = find_finger(warped)
    if rx is None:
        finger_x, finger_y = None, None
        smooth_x, smooth_y = None, None
    else:
        if smooth_x is None:
            smooth_x, smooth_y = float(rx), float(ry)
        else:
            smooth_x = 0.45 * rx + 0.55 * smooth_x
            smooth_y = 0.45 * ry + 0.55 * smooth_y
        finger_x, finger_y = int(smooth_x), int(smooth_y)

    # bubbles spawnen
    spawn_timer += dt
    if spawn_timer >= SPAWN_INTERVAL:
        spawn_timer = 0.0
        spawn_bubble()

    # bubbles bewegen
    for b in bubbles:
        if b['alive']:
            b['x'] += b['vx'] * dt
            if b['x'] < -BUBBLE_RADIUS:
                b['alive'] = False
                missed += 1

    # kollision checken
    if finger_x is not None:
        for b in bubbles:
            if b['alive']:
                dist = ((finger_x - b['x']) ** 2 + (finger_y - b['y']) ** 2) ** 0.5
                if dist < BUBBLE_RADIUS + 40:
                    b['alive'] = False
                    score += 1
                    pop_effects.append([b['x'], b['y'], 1.0])

    for e in pop_effects:
        e[2] -= dt * 3
    pop_effects[:] = [e for e in pop_effects if e[2] > 0]
    bubbles[:] = [b for b in bubbles if b['alive'] or b['x'] > -BUBBLE_RADIUS * 2]

    fb_w, fb_h = window.get_framebuffer_size()
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    current_img = cv2glet(cv2.resize(rgb, (fb_w, fb_h)))


@window.event
def on_draw():
    window.clear()

    if current_img:
        current_img.blit(0, 0)

    for b in bubbles:
        if b['alive']:
            wx, wy = cam_to_win(b['x'], b['y'])
            pyglet.shapes.Circle(wx, wy, int(BUBBLE_RADIUS * scale), color=(220, 50, 50)).draw()

    for e in pop_effects:
        wx, wy = cam_to_win(e[0], e[1])
        c = pyglet.shapes.Circle(wx, wy, int((BUBBLE_RADIUS + 30) * scale), color=(255, 240, 60))
        c.opacity = int(e[2] * 200)
        c.draw()

    if finger_x is not None:
        wx, wy = cam_to_win(finger_x, finger_y)
        pyglet.shapes.Circle(wx, wy, int(32 * scale), color=(50, 220, 50)).draw()

    pyglet.text.Label(f"Score: {score}  |  Missed: {missed}",
                      font_name='Arial', font_size=26,
                      x=10, y=10, color=(255, 220, 0, 255)).draw()

    if last_warp is None:
        pyglet.text.Label("Bitte Kamera auf das Board richten!",
                          font_name='Arial', font_size=22,
                          x=WIN_W // 2, y=WIN_H // 2,
                          anchor_x='center', color=(255, 80, 80, 230)).draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        cap.release()
        pyglet.app.exit()


@window.event
def on_close():
    cap.release()
    pyglet.app.exit()


pyglet.clock.schedule_interval(update, 1 / 30)
pyglet.app.run()
