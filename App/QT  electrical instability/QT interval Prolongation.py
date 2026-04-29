import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ QT Interval Prolongation
    ลักษณะ: ระยะห่างระหว่าง QRS complex กับ T-wave ยาวขึ้นกว่าปกติ
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. สร้างโครงสร้าง ECG (P, Q, R, S)
    y_base = (
        g(phase, 0.18, 0.030, 0.12) +  # P wave
        g(phase, 0.40, 0.012, -0.15) + # Q wave
        g(phase, 0.42, 0.008, 1.00) +  # R wave
        g(phase, 0.44, 0.012, -0.25)   # S wave
    )

    # 2. จำลองการยืดระยะ QT (Prolonged QT)
    # ปกติ T-wave จะอยู่ที่ phase ~0.70 
    # เราจะเลื่อนไปที่ phase ~0.85 เพื่อให้เห็นระยะที่ยาวขึ้นชัดเจน
    t_wave_prolonged = g(phase, 0.85, 0.080, 0.25) 
    
    # 3. เพิ่มความผิดปกติของ T-wave (เช่น ปลายมนหรือกว้างขึ้น)
    y_qt = y_base + t_wave_prolonged

    # 4. ใส่ Baseline wander และ Noise เล็กน้อย
    y_qt += 0.02 * np.sin(2 * np.pi * 0.3 * t)
    y_qt += 0.01 * np.random.randn(len(t))
    
    return y_qt