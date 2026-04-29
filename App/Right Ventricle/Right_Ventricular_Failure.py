import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Right Ventricular Failure (RVF)
    ลักษณะจำลอง:
    - Low Voltage overall (สัญญาณเตี้ยลงกว่าปกติ)
    - P wave สูงขึ้นเล็กน้อย (0.20 mV) จาก Right Atrial Enlargement
    - R wave ลดลง (0.75 mV) เพราะแรงบีบตัวลดลง
    - S wave กว้างและลึกพอประมาณ (-0.36 mV)
    - T wave inversion แบบตื้นๆ (-0.18 mV)
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. การสร้างโครงสร้างคลื่น RVF
    # P-wave: สูงขึ้นสะท้อนถึง RAE จากภาวะความดันในหัวใจห้องขวาสูง
    p_wave = g(phase, 0.18, 0.030, 0.20)
    
    # QRS Complex: สัญญาณเตี้ยลง (Reduced R) และกว้างขึ้นเล็กน้อย (Slurred S)
    qrs = (
        g(phase, 0.39, 0.015, -0.12) + # Q-wave เล็ก
        g(phase, 0.425, 0.014, 0.75) + # R-wave (เตี้ยลงกว่าปกติ)
        g(phase, 0.465, 0.025, -0.36)  # S-wave (ลึกและกว้างขึ้นเล็กน้อย)
    )
    
    # 2. ST-T Segment: Mild Inversion
    t_wave = g(phase, 0.72, 0.080, -0.18)

    y_rvf = p_wave + qrs + t_wave

    # 3. ใส่ความสมจริง: Low Voltage Feeling
    # ลดทอนสัญญาณโดยรวมลง 5% เพื่อจำลองประสิทธิภาพหัวใจที่แย่ลง
    y_rvf *= 0.95

    # 4. ใส่ Baseline wander และ Noise
    y_rvf += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_rvf += 0.01 * np.random.randn(len(t))
    
    return y_rvf