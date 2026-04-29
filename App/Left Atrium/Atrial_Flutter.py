import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fs = 250
duration = 30

t = np.arange(0, duration, 1/fs)

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

y_normal = ecg_waveform(t, rr)
y_normal += 0.03 * np.sin(2 * np.pi * 0.33 * t)
y_normal += 0.01 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# สัญญาณที่ 2 — Atrial Flutter (AFL)
# ลักษณะ: P wave ถูกแทนที่ด้วย flutter waves (sawtooth pattern)
# -----------------------------------------------------------------------
def afl_waveform(tt, rr):
    # Atrial Flutter (AFL) คลื่น flutter (sawtooth) ที่มีความถี่สูง
    flutter_freq = 0.250  # ความถี่ของการเต้น (250 bpm)
    phase = (tt * flutter_freq) % 1.0
    def g(mu, sig, amp):
        return amp * np.exp(-0.5*((phase - mu)/sig)**2)
    return (
        g(0.0, 0.030, 0.20) +  # Flutter waves (sawtooth pattern)
        g(0.5, 0.030, 0.20)    # Flutter waves (sawtooth pattern)
    )

y_afl = afl_waveform(t, rr)
y_afl += 0.03 * np.sin(2 * np.pi * 0.33 * t)
y_afl += 0.01 * np.random.randn(len(t))

# -----------------------------------------------------------------------
# Animation — 2 กราฟซ้อนกัน
# -----------------------------------------------------------------------
win_sec = 4.0
win     = int(win_sec * fs)
step    = 4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

line1, = ax1.plot([], [], lw=1.5, color='steelblue')
line2, = ax2.plot([], [], lw=1.5, color='red')  # เปลี่ยนเป็นสีแดงสำหรับ AFL

for ax, title, color, ylim in [
    (ax1, "Normal ECG (75 bpm)",         'steelblue', (-0.5,  1.30)),
    (ax2, "Atrial Flutter (AFL)",        'red',    (-0.5, 1.30)),  # สีแดงสำหรับ AFL
]:
    ax.set_ylim(ylim)
    ax.set_xlim(0, win_sec)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title, color=color, fontsize=11)
    ax.grid(True, alpha=0.3)

ax2.set_xlabel("Time (s)")
fig.suptitle("ECG Comparison — Normal vs AFL", fontsize=13, fontweight='bold')
fig.tight_layout()

frame_ref = [0]

def init():
    line1.set_data([], [])
    line2.set_data([], [])  # เพิ่มกราฟ AFL
    return line1, line2  # ลบ Polymorphic VT ออก

def update(_):
    i = frame_ref[0]
    frame_ref[0] += step
    if i + win >= len(t):
        frame_ref[0] = 0
        i = 0

    tt = t[i:i+win] - t[i]
    line1.set_data(tt, y_normal[i:i+win])
    line2.set_data(tt, y_afl[i:i+win])  # เพิ่มกราฟ AFL
    return line1, line2  # ลบ Polymorphic VT ออก

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    interval=30, blit=False, repeat=True, cache_frame_data=False
)

plt.show()