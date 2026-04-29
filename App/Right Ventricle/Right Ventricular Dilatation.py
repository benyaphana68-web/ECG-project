import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Right Ventricular Dilatation (RVD)
    ลักษณะจำลอง (Mild RV Overload Pattern):
    - P wave สูงขึ้น (0.22 mV) จาก Right Atrial Enlargement (RAE) เบื้องต้น
    - QRS กว้างขึ้นเล็กน้อย และมี S-wave ลึกขึ้น (-0.42 mV)
    - T wave inversion ตื้น ๆ (-0.20 mV) สะท้อนถึง mild strain
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. การสร้างคลื่น RVD
    # P-wave: สูงแหลมขึ้นกว่าปกติเล็กน้อย
    p_wave = g(phase, 0.18, 0.028, 0.22)
    
    # QRS Complex: เน้น S-wave ที่ลึกและกว้างขึ้นเล็กน้อย
    qrs = (
        g(phase, 0.39, 0.015, -0.14) + # Q-wave
        g(phase, 0.425, 0.014, 0.95) + # R-wave (สูงแต่ไม่เกิน 1.0 เพื่อไม่ให้เป็น RVH)
        g(phase, 0.470, 0.025, -0.42)  # S-wave ลึกขึ้น (เป็นจุดสังเกตหลัก)
    )
    
    # 2. ST-T Segment: T-wave หัวกลับแบบตื้น (Mild Inversion)
    t_wave = g(phase, 0.72, 0.075, -0.20)

    y_rvd = p_wave + qrs + t_wave

    # 3. ใส่สัญญาณรบกวน (Noise) และ Baseline wander
    y_rvd += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_rvd += 0.01 * np.random.randn(len(t))
    
    return y_rvd