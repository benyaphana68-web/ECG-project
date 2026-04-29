import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# สร้างสัญญาณ ECG แบบจำลอง (ไม่ใช่ medical-grade) --------------------------
fs       = 250    # sampling rate (Hz)
duration = 8      # seconds

t = np.arange(0, duration, 1/fs)

hr = 75     # bpm
rr = 60/hr  # seconds/beat

# สร้าง P-QRS-T แบบง่ายด้วย gaussian หลายลูก
def ecg_waveform(tt, rr):
    phase = (tt % rr) / rr

    def g(mu, sig, amp):
        return amp * np.exp(-0.5*((phase - mu)/sig)**2)

    y = (
        g(0.18, 0.030,  0.12) +    # P
        g(0.40, 0.012, -0.15) +    # Q
        g(0.42, 0.008,  1.00) +    # R
        g(0.44, 0.012, -0.25) +    # S
        g(0.70, 0.060,  0.30)      # T
    )
    return y

y = ecg_waveform(t, rr)

y += 0.03*np.sin(2*np.pi*0.33*t)   # baseline wander
y += 0.01*np.random.randn(len(t))   # noise

# ทำ animation แบบ "หน้าต่างเลื่อน" ----------------------------------------
win_sec = 3.0
win = int(win_sec*fs)
step = 6  # ยิ่งมากยิ่งเร็ว/กระโดดมาก

fig, ax = plt.subplots(figsize=(10, 3))
line, = ax.plot([], [], lw=2, color='steelblue')
ax.set_ylim(-0.5, 1.25)
ax.set_xlim(0, win_sec)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Animated ECG (simulated)")
ax.grid(True, alpha=0.3)
fig.tight_layout()

def init():
    line.set_data([], [])
    return (line,)

def update(frame):
    i = frame * step
    if i + win >= len(t):
        i = 0
    tt = t[i:i+win] - t[i]
    yy = y[i:i+win]
    line.set_data(tt, yy)
    return (line,)

frames = (len(t) - win) // step
ani = animation.FuncAnimation(
    fig, update, init_func=init, frames=frames, interval=30, blit=True, repeat=True
)

plt.show()
# -------------------------------------------------------------------------