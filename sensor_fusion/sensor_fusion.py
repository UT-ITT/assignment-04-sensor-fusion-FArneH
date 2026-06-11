import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import pyglet.shapes
import sys
from PIL import Image
from DIPPID import SensorUDP

# DIPPID verbinden
sensor = SensorUDP(5700)

# marker ID vom handy (DIPPID app zeigt ID 23 an)
MARKER_ID = 23

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

cap = cv2.VideoCapture(video_id)
ret, frame0 = cap.read()
if not ret:
    print("kamera fehler")
    sys.exit(1)

CAM_H, CAM_W = frame0.shape[:2]

screen = pyglet.display.get_display().get_default_screen()
scale = min((screen.width * 0.9) / CAM_W, (screen.height * 0.85) / CAM_H, 1.5)
WIN_W = int(CAM_W * scale)
WIN_H = int(CAM_H * scale)
window = pyglet.window.Window(WIN_W, WIN_H, "Sensor Fusion")

# aruco detector mit lockeren parametern damit marker besser erkannt wird
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
params = aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 53
params.adaptiveThreshWinSizeStep = 4
params.minMarkerPerimeterRate = 0.02
params.errorCorrectionRate = 1.0
detector = aruco.ArucoDetector(aruco_dict, params)

last_warp = None
current_img = None
cam_x = None
cam_y = None
pred_x = None
pred_y = None
alpha = 0.5
ACCEL_SCALE = 250


def cv2glet(img):
    rows, cols, ch = img.shape
    raw = Image.fromarray(img).tobytes()
    return pyglet.image.ImageData(cols, rows, 'RGB', raw, pitch=-cols * ch)


def cam_to_win(x, y):
    wx = int(x / CAM_W * window.width)
    wy = int((CAM_H - y) / CAM_H * window.height)
    return wx, wy


def get_warp(corners):
    centers = np.array([c[0].mean(axis=0) for c in corners])
    s = centers.sum(axis=1)
    d = np.diff(centers, axis=1).flatten()
    src = np.array([centers[np.argmin(s)], centers[np.argmin(d)],
                    centers[np.argmax(s)], centers[np.argmax(d)]], dtype=np.float32)
    dst = np.array([[0, 0], [CAM_W-1, 0], [CAM_W-1, CAM_H-1], [0, CAM_H-1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def on_button1(data):
    global pred_x, pred_y
    if data == 1:
        pred_x = cam_x
        pred_y = cam_y


sensor.register_callback('button_1', on_button1)


def update(dt):
    global current_img, last_warp, cam_x, cam_y, pred_x, pred_y

    ret, frame = cap.read()
    if not ret:
        return

    # board erkennen (marker ID 23 ausschliessen)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_corners, all_ids, _ = detector.detectMarkers(gray)
    board_corners = []
    if all_ids is not None:
        for i, mid in enumerate(all_ids.flatten()):
            if mid != MARKER_ID:
                board_corners.append(all_corners[i])
    if len(board_corners) >= 4:
        last_warp = get_warp(board_corners[:4])

    if last_warp is not None:
        warped = cv2.warpPerspective(frame, last_warp, (CAM_W, CAM_H))
    else:
        warped = frame

    # handy marker in gewarptem bild suchen
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    corners2, ids2, _ = detector.detectMarkers(gray_w)
    cam_x, cam_y = None, None
    if ids2 is not None:
        for i, mid in enumerate(ids2.flatten()):
            if mid == MARKER_ID:
                c = corners2[i][0].mean(axis=0)
                cam_x, cam_y = int(c[0]), int(c[1])
                break

    # komplementaer filter
    if cam_x is not None:
        acc = sensor.get_value('accelerometer')
        if pred_x is None:
            pred_x, pred_y = float(cam_x), float(cam_y)
        else:
            if acc is not None:
                pred_x += acc['x'] * ACCEL_SCALE * dt
                pred_y += acc['y'] * ACCEL_SCALE * dt
            pred_x = alpha * cam_x + (1 - alpha) * pred_x
            pred_y = alpha * cam_y + (1 - alpha) * pred_y

    fb_w, fb_h = window.get_framebuffer_size()
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    current_img = cv2glet(cv2.resize(rgb, (fb_w, fb_h)))


@window.event
def on_draw():
    window.clear()
    if current_img:
        current_img.blit(0, 0)

    # roter punkt = kamera
    if cam_x is not None:
        wx, wy = cam_to_win(cam_x, cam_y)
        pyglet.shapes.Circle(wx, wy, int(14 * scale), color=(220, 50, 50)).draw()

    # gruener punkt = komplementaer filter vorhersage
    if pred_x is not None:
        wx, wy = cam_to_win(int(pred_x), int(pred_y))
        pyglet.shapes.Circle(wx, wy, int(14 * scale), color=(50, 220, 50)).draw()

    pyglet.text.Label(f"alpha: {alpha:.1f}  (Pfeiltasten,  Button 1 = Reset,  Q = Beenden)",
                      font_name='Arial', font_size=16,
                      x=10, y=10, color=(255, 255, 255, 220)).draw()

    if last_warp is None:
        pyglet.text.Label("Bitte Kamera auf das Board richten!",
                          font_name='Arial', font_size=22,
                          x=WIN_W // 2, y=WIN_H // 2,
                          anchor_x='center', color=(255, 80, 80, 230)).draw()


@window.event
def on_key_press(symbol, modifiers):
    global alpha
    if symbol == pyglet.window.key.Q:
        cap.release()
        sensor.disconnect()
        pyglet.app.exit()
    elif symbol == pyglet.window.key.UP or symbol == pyglet.window.key.RIGHT:
        alpha = min(1.0, round(alpha + 0.1, 1))
        print(f"alpha = {alpha}")
    elif symbol == pyglet.window.key.DOWN or symbol == pyglet.window.key.LEFT:
        alpha = max(0.0, round(alpha - 0.1, 1))
        print(f"alpha = {alpha}")


@window.event
def on_close():
    cap.release()
    sensor.disconnect()
    pyglet.app.exit()


pyglet.clock.schedule_interval(update, 1 / 30)
pyglet.app.run()
