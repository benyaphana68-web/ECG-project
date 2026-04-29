import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fs       = 250
duration = 30
t        = np.arange(0, duration, 1/fs)

# -----------------------------------------------------------------------
# สัญญาณที่ 1 — ECG ปกติ (75 bpm)
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

y_normal  = ecg_waveform(t, rr)
y_normal += 0.03 * np.sin(2 * np.pi * 0.33 * t)
y_normal += 0.01 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# สัญญาณที่ 2 — Ventricular Hypertrophy (LVH)
# -----------------------------------------------------------------------
np.random.seed(11)

y_lvh = np.zeros(len(t))

rr_lvh = rr  # จังหวะยังปกติ

for bt in np.arange(0, duration, rr_lvh):

    # amplitude สูง (ผนังหนา)
    R_amp = np.random.uniform(1.8, 2.8)

    # QRS กว้างขึ้นเล็กน้อย
    qrs_w = np.random.uniform(0.020, 0.040)

    # P wave
    y_lvh += 0.12 * np.exp(-0.5*((t-(bt+0.18*rr))/0.030)**2)

    # Q
    y_lvh += -0.25 * R_amp * np.exp(-0.5*((t-(bt+0.40*rr))/qrs_w)**2)

    # R สูงมาก
    y_lvh += R_amp * np.exp(-0.5*((t-(bt+0.42*rr))/0.010)**2)

    # S ลึก
    y_lvh += -0.35 * R_amp * np.exp(-0.5*((t-(bt+0.44*rr))/qrs_w)**2)

    # ST depression (strain)
    y_lvh += -0.1 * np.exp(-0.5*((t-(bt+0.55*rr))/0.040)**2)

    # T wave กลับหัว
    y_lvh += -0.4 * np.exp(-0.5*((t-(bt+0.70*rr))/0.060)**2)

# baseline + noise
y_lvh += 0.02 * np.sin(2 * np.pi * 0.3 * t)
y_lvh += 0.01 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------
win_sec = 4.0
win     = int(win_sec * fs)
step    = 4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

line1, = ax1.plot([], [], lw=1.5, color='steelblue')
line2, = ax2.plot([], [], lw=1.5, color='crimson')

for ax, title, color, ylim in [
    (ax1, "Normal ECG (75 bpm)", 'steelblue', (-0.5,  1.30)),
    (ax2, "Ventricular Hypertrophy (LVH)", 'crimson', (-3.00, 3.00)),
]:
    ax.set_ylim(ylim)
    ax.set_xlim(0, win_sec)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title, color=color, fontsize=11)
    ax.grid(True, alpha=0.3)

ax2.set_xlabel("Time (s)")
fig.suptitle("ECG Comparison — Normal vs Ventricular Hypertrophy (LVH)",
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
    line2.set_data(tt, y_lvh[i:i+win])
    return line1, line2

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    interval=30, blit=False, repeat=True, cache_frame_data=False
)

plt.show()