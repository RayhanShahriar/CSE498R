import cv2
import numpy as np
import math
import time
import pickle  # <-- added for saving motion path

def combineTrackers(oldPoints, newPoints, status):
    threshold = 5
    extraFeatures = []
    for i in range(len(newPoints)):
        new = True
        for j in range(len(oldPoints)):
            if status[j] == 1:
                if math.hypot(newPoints[i][0] - oldPoints[j][0], newPoints[i][1] - oldPoints[j][1]) < threshold:
                    new = False
        if new:
            extraFeatures += [newPoints[i].tolist()]
    if len(extraFeatures) != 0:
        featureList = np.concatenate((oldPoints, np.asarray(extraFeatures)))
    else:
        featureList = oldPoints
    return featureList

# Load the video
video_path = 'corridor.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (width, height))

# Optical flow params
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=2, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
colourList = [np.random.random(size=3) * 256 for _ in range(5000)]

# First frame and features
ret, prev_frame = cap.read()
if not ret:
    print("Error reading first frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_feat = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
if prev_feat is None or len(prev_feat) == 0:
    print("No features in first frame.")
    exit()
prev_feat = np.reshape(prev_feat, (-1, 1, 2)).astype(np.float32)

mask = np.zeros_like(prev_frame)
cv2.namedWindow("sparse optical flow", cv2.WINDOW_NORMAL)
cv2.resizeWindow("sparse optical flow", 800, 600)

frame_idx = 0
t_start = time.time()

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_feat is None or len(prev_feat) == 0:
        prev_feat = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_feat is None or len(prev_feat) == 0:
            frame_idx += 1
            continue
        prev_feat = np.reshape(prev_feat, (-1, 1, 2)).astype(np.float32)
    else:
        prev_feat = np.reshape(prev_feat, (-1, 1, 2)).astype(np.float32)

    nxt, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_feat, None, **lk_params)
    if nxt is None or status is None or len(nxt) == 0:
        frame_idx += 1
        continue

    good_old = prev_feat[:, 0, :]
    good_new = nxt[:, 0, :]

    for j in range(len(good_new)):
        new_coord = tuple(map(int, good_new[j]))
        old_coord = tuple(map(int, good_old[j]))
        frame = cv2.circle(frame, new_coord, 3, colourList[j], -1)

    if len(good_new) > 14:
        mask = cv2.line(mask, tuple(map(int, good_new[14])), tuple(map(int, good_old[14])), (0, 255, 0), 2)

    if (frame_idx % int(fps)) == 0:
        new_trackers = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if new_trackers is not None:
            good_new = combineTrackers(np.rint(good_new), new_trackers[:, 0, :], status)
            good_new = np.reshape(good_new, (-1, 1, 2)).astype(np.float32)

    output = cv2.add(frame, mask)
    cv2.imshow("sparse optical flow", output)
    out.write(output)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    prev_feat = good_new.reshape(-1, 1, 2)
    prev_gray = gray.copy()
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Tracking complete. Video saved as 'tracked_output.mp4'")
print(f"⏱ Total runtime: {time.time() - t_start:.2f} seconds")

# ============ SAVE MOTION PATH FOR SIMULATION ============

cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)

motion_path = [(0.0, 0.0, 0.0)]
x, y = 0.0, 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, prev_pts, None, **lk_params)

    if next_pts is None or status is None:
        break

    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    dx, dy = 0.0, 0.0
    count = 0
    for p1, p2 in zip(good_prev, good_next):
        dx += p2[0] - p1[0]
        dy += p2[1] - p1[1]
        count += 1

    if count > 0:
        dx /= count
        dy /= count
        x += dx
        y += dy
        bearing = math.atan2(dy, dx)
        motion_path.append((x, y, bearing))

    gray_prev = gray.copy()
    prev_pts = good_next.reshape(-1, 1, 2)

cap.release()

with open("camera_motion.pkl", "wb") as f:
    pickle.dump(motion_path, f)
print("📦 Camera path saved to camera_motion.pkl")
