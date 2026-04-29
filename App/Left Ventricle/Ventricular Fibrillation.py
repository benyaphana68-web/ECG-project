import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
# สัญญาณที่ 2 — Ventricular Fibrillation (VF)
# ลักษณะ: ไม่มี QRS ชัดเจน คลื่นสั่นพร่า ความถี่ 3–10 Hz
# amplitude ไม่สม่ำเสมอ สุ่มโกลาหล ไม่มีรูปแบบ
# -----------------------------------------------------------------------
np.random.seed(99)

# VF = คลื่นความถี่หลายๆ ความถี่ซ้อนกัน แบบสุ่ม
y_vf = np.zeros(len(t))

# เพิ่มคลื่น sine หลายความถี่ในช่วง 3–10 Hz (ช่วง VF จริง)
for freq in np.arange(3.0, 10.5, 0.5):
    amp_rand   = np.random.uniform(0.05, 0.25)
    phase_rand = np.random.uniform(0, 2 * np.pi)
    y_vf += amp_rand * np.sin(2 * np.pi * freq * t + phase_rand)

# เพิ่ม amplitude modulation — ให้ขึ้นลงไม่สม่ำเสมอ
mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t) + 0.3 * np.sin(2 * np.pi * 1.3 * t)
y_vf *= mod

# เพิ่ม noise แบบสุ่มโกลาหล
y_vf += 0.08 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# Animation — 2 กราฟซ้อนกัน
# -----------------------------------------------------------------------
win_sec = 4.0
win     = int(win_sec * fs)
step    = 4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

line1, = ax1.plot([], [], lw=1.5, color='steelblue')
line2, = ax2.plot([], [], lw=1.5, color='crimson')

for ax, title, color, ylim in [
    (ax1, "Normal ECG (75 bpm)",              'steelblue', (-0.5,  1.30)),
    (ax2, "Ventricular Fibrillation (VF)",    'crimson',   (-1.20, 1.20)),
]:
    ax.set_ylim(ylim)
    ax.set_xlim(0, win_sec)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title, color=color, fontsize=11)
    ax.grid(True, alpha=0.3)

ax2.set_xlabel("Time (s)")
fig.suptitle("ECG Comparison — Normal vs Ventricular Fibrillation",
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
    line2.set_data(tt, y_vf[i:i+win])
    return line1, line2

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    interval=30, blit=False, repeat=True, cache_frame_data=False
)

plt.show()