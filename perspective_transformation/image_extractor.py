import sys
import cv2
import numpy as np

# usage: python image_extractor.py <input_image> <output_path> <width> <height>
# click 4 corners, press S to save, ESC to reset, Q to quit

if len(sys.argv) != 5:
    print("Usage: python image_extractor.py <input_image> <output_path> <width> <height>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]
out_w = int(sys.argv[3])
out_h = int(sys.argv[4])

img = cv2.imread(input_path)
if img is None:
    print("Could not load image:", input_path)
    sys.exit(1)

WINDOW_NAME = "Image Extractor"
points = []
warped = None
show_result = False


def order_points(pts):
    # sort points into top-left, top-right, bottom-right, bottom-left
    # based on sum and difference of coordinates
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.array([
        pts[np.argmin(s)],   # top-left has smallest sum
        pts[np.argmin(d)],   # top-right has smallest diff
        pts[np.argmax(s)],   # bottom-right has largest sum
        pts[np.argmax(d)],   # bottom-left has largest diff
    ], dtype=np.float32)
    return ordered


def draw_points(image):
    # draw the selected points and lines between them on a copy of the image
    vis = image.copy()
    for i, pt in enumerate(points):
        cv2.circle(vis, pt, 6, (0, 255, 0), -1)
        cv2.putText(vis, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(vis, points[i - 1], points[i], (0, 200, 255), 2)
        if len(points) == 4:
            cv2.line(vis, points[3], points[0], (0, 200, 255), 2)
    hint = f"Select corners: {len(points)}/4  |  ESC = reset  |  Q = quit"
    cv2.putText(vis, hint, (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, hint, (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return vis


def mouse_callback(event, x, y, _flags, _param):
    global warped, show_result

    if show_result:
        return
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(points) >= 4:
        return

    points.append((x, y))
    cv2.imshow(WINDOW_NAME, draw_points(img))

    # once we have 4 points, do the perspective transform
    if len(points) == 4:
        src_pts = order_points(points)
        dst_pts = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (out_w, out_h))

        # show the result with a hint
        result_vis = warped.copy()
        cv2.putText(result_vis, "S = save  |  ESC = back  |  Q = quit",
                    (10, result_vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(result_vis, "S = save  |  ESC = back  |  Q = quit",
                    (10, result_vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.imshow(WINDOW_NAME, result_vis)
        show_result = True


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
cv2.imshow(WINDOW_NAME, draw_points(img))

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q') or key == ord('Q'):
        break

    elif key == 27:  # ESC
        if points or show_result:
            # reset everything
            points.clear()
            warped = None
            show_result = False
            cv2.imshow(WINDOW_NAME, draw_points(img))
        else:
            break  # nothing to reset, just quit

    elif key == ord('s') or key == ord('S'):
        if show_result and warped is not None:
            cv2.imwrite(output_path, warped)
            print("Saved to", output_path)

    # stop loop if window was closed
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

cv2.destroyAllWindows()
