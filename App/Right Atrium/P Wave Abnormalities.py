import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ P-Wave Abnormalities
    ลักษณะจำลอง: Notched P wave (P mitrale) - P wave กว้างขึ้นและมี 2 ยอด
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. สร้าง P-wave ที่ผิดปกติ (Notched / Bifid P wave)
    # ผสม Gaussian 2 ลูกใกล้กันเพื่อให้เกิดรอยหยัก (Notch)
    p_wave = (
        g(phase, 0.140, 0.025, 0.18) + # ยอดที่ 1 (Right atrium)
        g(phase, 0.200, 0.030, 0.22)   # ยอดที่ 2 (Left atrium)
    )

    # 2. สร้างโครงสร้าง ECG ส่วนที่เหลือ (QRS และ T wave ยังคงเดิม)
    y_pwave = (
        p_wave +
        g(phase, 0.40, 0.012, -0.15) + # Q
        g(phase, 0.42, 0.008, 1.00) +  # R
        g(phase, 0.44, 0.012, -0.25) + # S
        g(phase, 0.70, 0.060, 0.30)    # T
    )

    # 3. ใส่สัญญาณรบกวน (Noise) และ Baseline wander ตามที่คุณตั้งค่าไว้
    y_pwave += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_pwave += 0.01 * np.random.randn(len(t))
    
    return y_pwave