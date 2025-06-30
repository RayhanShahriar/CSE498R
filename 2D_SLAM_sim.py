# ✅ Updated 2D_SLAM_Sim.py with improvements to velocity estimation, angle smoothing, and layout fixes

import numpy as np
import math
import pickle
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from filterpy.stats import plot_covariance_ellipse

# === Load motion from optical_flow.py ===
with open("camera_motion.pkl", "rb") as f:
    camera_path = pickle.load(f)

# === Initialize random targets ===
nT = 50
np.random.seed(0)
targets = [[np.random.uniform(-400, 400), np.random.uniform(0, 800)] for _ in range(nT)]

# === Generate simulated angle measurements ===
def generate_measurements(xp, yp, bearing, targets):
    measurements = []
    for T in targets:
        err = 0.05 * np.random.randn()
        ang = math.atan2((T[1] - yp), (T[0] - xp)) + bearing
        measurements.append([ang + err])
    return np.asarray(measurements)

# === Jacobian ===
def XJacobian(x):
    x = x.flatten()
    xp, yp, vp, bear = x[0], x[1], x[2], x[3]
    nT = int((len(x) - 4) / 2)
    nS = len(x)
    out = np.zeros((nT, nS))
    for i in range(nT):
        tIndx = 4 + 2 * i
        dx = x[tIndx] - xp
        dy = x[tIndx + 1] - yp
        denom = dx**2 + dy**2
        if denom == 0: denom = 1e-6
        out[i, tIndx] = dy / denom
        out[i, tIndx + 1] = -dx / denom
        out[i, 0] = -dy / denom
        out[i, 1] = dx / denom
        out[i, 3] = 1
    return out

# === Measurement function ===
def hx(x):
    x = x.flatten()
    xp, yp, vp, bear = x[0], x[1], x[2], x[3]
    nT = int((len(x) - 4) / 2)
    out = []
    for i in range(nT):
        tIndx = 4 + 2 * i
        dx = x[tIndx] - xp
        dy = x[tIndx + 1] - yp
        angle = math.atan2(dy, dx) + bear
        out.append([angle])
    return np.asarray(out)

# === Residual for angle wrapping ===
def residual(a, b):
    y = a - b
    y = y % (2 * np.pi)
    y[y > np.pi] -= 2 * np.pi
    return y

# === EKF Setup ===
ekf = ExtendedKalmanFilter(dim_x=4 + 2 * nT, dim_z=nT)
np_targets = np.asarray(targets) + 75 * np.random.randn(nT, 2)

init_x = camera_path[0][0] + 5
init_y = camera_path[0][1] + 7
init_v = 0.0
init_b = camera_path[0][2] + 1.5

ekf.x = np.array([init_x, init_y, init_v, init_b])
ekf.x = np.append(ekf.x, np_targets)
ekf.x = ekf.x.reshape(-1, 1)

dt = 0.05
ekf.F = np.eye(4 + 2 * nT)
ekf.F[1, 2] = dt

# Measurement and process noise tuning
ekf.R = np.eye(nT) * 0.05
ekf.Q = np.eye(4 + 2 * nT) * 1e-7
ekf.Q[0, 0], ekf.Q[1, 1], ekf.Q[2, 2], ekf.Q[3, 3] = 1e-2, 1e-2, 1e-3, 1e-4  # Increased position/velocity process noise
ekf.P *= 1500

# === Run EKF simulation ===
cameraTrk, predictionTrk, updateCovarTrk = [], [], []
MAng, TargetX, TargetX_Error = [], [], []

# Smoothed velocity (from displacement)
prev_x, prev_y = camera_path[0][0], camera_path[0][1]

for idx, (x_pos, y_pos, bearing) in enumerate(camera_path):
    dx = x_pos - prev_x
    dy = y_pos - prev_y
    velocity = np.hypot(dx, dy) / dt
    bearing = (0.8 * ekf.x[3, 0]) + (0.2 * bearing)  # smooth bearing

    ekf.x[0] = x_pos
    ekf.x[1] = y_pos
    ekf.x[2] = velocity
    ekf.x[3] = bearing

    z = generate_measurements(x_pos, y_pos, bearing, targets)

    cameraTrk.append((x_pos, y_pos, velocity, bearing))
    MAng.append(z[0][0])
    TargetX.append((ekf.x[4:]))
    TargetX_Error.append(ekf.x[4:].flatten() - np.asarray(targets).flatten())

    ekf.update(z, XJacobian, hx, residual=residual)
    predictionTrk.append((ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0]))
    updateCovarTrk.append(ekf.P.copy())
    ekf.predict()

    prev_x, prev_y = x_pos, y_pos

# === Convert results ===
cameraTrk = np.array(cameraTrk)
predictionTrk = np.array(predictionTrk)
updateCovarTrk = np.array(updateCovarTrk)
MAng = np.array(MAng)
TargetX = np.array(TargetX)
TargetX_Error = np.array(TargetX_Error)

# === Static Plots ===
xaxis = np.arange(0, len(cameraTrk) * dt, dt)
titles = ['X Position', 'Y Position', 'Y Velocity']
ylabels = ['X-Position (m)', 'Y-Position (m)', 'Y-Velocity (m/s)', 'Bearing (radians)']
fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

for plot in range(3):
    axs[0, plot].set_title(titles[plot])
    axs[0, plot].set_xlabel('Time (s)')
    axs[0, plot].set_ylabel(ylabels[plot])
    axs[0, plot].plot(xaxis, cameraTrk[:, plot], 'k', label='Ground Truth')
    axs[0, plot].plot(xaxis, predictionTrk[:, plot], 'r', label='EKF')
    axs[0, plot].legend()

axs[1, 0].set_title('Camera Trajectory')
axs[1, 0].set_xlabel('X-Position (m)')
axs[1, 0].set_ylabel('Y-Position (m)')
axs[1, 0].plot(cameraTrk[:, 0], cameraTrk[:, 1], 'ko', markersize=1, markevery=5)
axs[1, 0].plot(predictionTrk[:, 0], predictionTrk[:, 1], 'ro', markersize=1, markevery=5)
for t in targets:
    axs[1, 0].plot(t[0], t[1], 'bx', markersize=4, alpha=0.7)

axs[1, 1].set_title('X Target Error')
axs[1, 2].set_title('Y Target Error')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 2].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('X Error (m)')
axs[1, 2].set_ylabel('Y Error (m)')

xavg = 0
yavg = 0
for i in range(0, nT * 2, 2):
    axs[1, 1].plot(xaxis, TargetX_Error[:, i])
    xavg += TargetX_Error[-1, i]
    axs[1, 2].plot(xaxis, TargetX_Error[:, i + 1])
    yavg += TargetX_Error[-1, i + 1]

print("Average X Error:", xavg / nT)
print("Average Y Error:", yavg / nT)

# === Covariance Ellipses ===
cov_fig = plt.figure()
plt.plot(cameraTrk[:, 0], cameraTrk[:, 1], 'ko', markersize=1, markevery=10, alpha=0.3)
plt.title('Camera covariance plot')
plt.xlabel('X-Position (m)')
plt.ylabel('Y-Position (m)')
for t in targets:
    plt.plot(t[0], t[1], 'bx', markersize=4)
for i in range(0, predictionTrk.shape[0], 500):
    plot_covariance_ellipse(
        (predictionTrk[i, 0], predictionTrk[i, 1]), updateCovarTrk[i, 0:2, 0:2],
        std=6, facecolor='g', alpha=0.8)

# === Animate ===
TargetX = np.reshape(TargetX, (TargetX.shape[0], nT, 2))[::10]
cameraTrk = cameraTrk[::10]
predictionTrk = predictionTrk[::10]

fig_anim = plt.figure()
fig_anim.set_size_inches(6, 6, True)
ax_anim = plt.axes(xlim=(-500, 500), ylim=(0, 800))
ax_anim.set_title('Camera Trajectory Animation')
ax_anim.set_xlabel('X-Position (m)')
ax_anim.set_ylabel('Y-Position (m)')

ground_truth, = ax_anim.plot([], [], 'ko', markersize=1)
kalman_track, = ax_anim.plot([], [], 'ro', markersize=1)
target_pred, = ax_anim.plot([], [], 'go', markersize=4, alpha=0.7)
lines = [plt.plot([], [], 'yo-', animated=True, markersize=0)[0] for _ in range(nT)]
for t in targets:
    ax_anim.plot(t[0], t[1], 'bx', markersize=4, alpha=0.7)

def animate(i):
    ground_truth.set_data(cameraTrk[:i, 0], cameraTrk[:i, 1])
    kalman_track.set_data(predictionTrk[:i, 0], predictionTrk[:i, 1])
    target_pred.set_data(TargetX[i, :, 0], TargetX[i, :, 1])
    for j in range(nT):
        lines[j].set_data([TargetX[i, j, 0], targets[j][0]], [TargetX[i, j, 1], targets[j][1]])

anim = FuncAnimation(fig_anim, animate, frames=len(cameraTrk)-1, interval=0)

anim.save('Seed_0_({}targets).mp4'.format(nT), fps=60, bitrate=1000)
plt.close()
print("Video Outputted.")
plt.show()
