import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

fs       = 250
duration = 30

t = np.arange(0, duration, 1/fs)

# -----------------------------------------------------------------------
# สัญญาณที่ 1 — ECG ปกติ (75 bpm คงที่)
# -----------------------------------------------------------------------
hr = 75
rr = 60 / hr

def ecg_waveform(tt, rr):
    phase = (tt % rr) / rr
    def g(mu, sig, amp):
        return amp * np.exp(-0.5*((phase - mu)/sig)**2)
    return (
        g(0.18, 0.030,  0.12) +
        g(0.40, 0.012, -0.15) +
        g(0.42, 0.008,  1.00) +
        g(0.44, 0.012, -0.25) +
        g(0.70, 0.060,  0.30)
    )

y_normal = ecg_waveform(t, rr)
y_normal += 0.03 * np.sin(2 * np.pi * 0.33 * t)
y_normal += 0.01 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# สัญญาณที่ 2 — Atrial Fibrillation (AF)
# ลักษณะ:
#   1. ไม่มี P wave — แทนด้วยคลื่นสั่นเล็ก (fibrillatory baseline)
#   2. RR interval ไม่สม่ำเสมอ (irregularly irregular)
#   3. QRS ยังมีรูปร่างปกติ แต่เกิดในเวลาสุ่ม
# -----------------------------------------------------------------------
def gaussian(tt, center, sig, amp):
    return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

np.random.seed(42)
random.seed(42)

y_af = np.zeros(len(t))

# --- Fibrillatory baseline (แทน P wave) ---
# คลื่นเล็กๆ ความถี่ 350–600 Hz จำลองเป็น 4–8 Hz สุ่มซ้อนกัน
for freq in np.arange(4.0, 9.0, 0.7):
    amp_f   = np.random.uniform(0.02, 0.06)
    phase_f = np.random.uniform(0, 2 * np.pi)
    y_af += amp_f * np.sin(2 * np.pi * freq * t + phase_f)

# --- QRS ปกติ แต่ RR ไม่สม่ำเสมอ (irregularly irregular) ---
bt = 0
while bt < duration:
    # RR สุ่มในช่วง 0.5–1.1 s (AF มักเต้น 60–120 bpm)
    RR = random.uniform(0.50, 1.10)
    bt += RR

    # QRS ปกติ (ไม่มี P wave)
    y_af += gaussian(t, bt + 0.07, 0.010, -0.15)   # Q
    y_af += gaussian(t, bt + 0.09, 0.007,  1.00)   # R
    y_af += gaussian(t, bt + 0.11, 0.010, -0.25)   # S
    y_af += gaussian(t, bt + 0.27, 0.050,  0.28)   # T

y_af += 0.01 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# Animation — 2 กราฟซ้อนกัน
# -----------------------------------------------------------------------
win_sec = 4.0
win     = int(win_sec * fs)
step    = 4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

line1, = ax1.plot([], [], lw=1.5, color='steelblue')
line2, = ax2.plot([], [], lw=1.5, color='tomato')

for ax, title, color, ylim in [
    (ax1, "Normal ECG (75 bpm)",        'steelblue', (-0.5,  1.30)),
    (ax2, "Atrial Fibrillation (AF)",   'tomato',    (-0.5,  1.30)),
]:
    ax.set_ylim(ylim)
    ax.set_xlim(0, win_sec)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title, color=color, fontsize=11)
    ax.grid(True, alpha=0.3)

ax2.set_xlabel("Time (s)")
fig.suptitle("ECG Comparison — Normal vs Atrial Fibrillation",
             fontsize=13, fontweight='bold')
fig.tight_layout()

frame_ref = [0]

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update(_):
    i = frame_ref[0]
    frame_ref[0] += step
    if i + win >= len(t):
        frame_ref[0] = 0
        i = 0
    tt = t[i:i+win] - t[i]
    line1.set_data(tt, y_normal[i:i+win])
    line2.set_data(tt, y_af[i:i+win])
    return line1, line2

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    interval=30, blit=False, repeat=True, cache_frame_data=False
)

plt.show()