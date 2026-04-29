import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ VT Aneurysm (Ventricular Tachycardia associated with Aneurysm)
    ลักษณะ: Wide QRS, หัวใจเต้นเร็ว (Tachycardia) และลักษณะคลื่นที่กว้างและสูง
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    # 1. ตั้งค่าพื้นฐาน (เต้นเร็ว ~150 bpm)
    hr_vt = 150
    rr_vt = 60 / hr_vt
    phase = (t % rr_vt) / rr_vt
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 2. สร้าง Wide QRS Complex 
    # ปรับความกว้าง (sig) ให้มากกว่าปกติ และเพิ่ม Amplitude (amp) 
    # เพื่อให้เห็นลักษณะ aneurysm-related VT
    
    # QRS ที่กว้างและสูง (Wide & Bizarre QRS)
    y_vt = (
        g(phase, 0.10, 0.015, -0.25) + # Q
        g(phase, 0.13, 0.035,  1.65) + # R (Wide and Tall)
        g(phase, 0.18, 0.025, -0.40) + # S
        g(phase, 0.35, 0.070, -0.45)   # T (สวนทางกับ QRS)
    )

    # 3. เพิ่มความไม่สม่ำเสมอ (Irregular Rhythm) เล็กน้อยตาม Logic เดิมของคุณ
    # แต่ปรับให้คงที่ในแต่ละเฟรม
    irregular = 0.15 * np.sin(2 * np.pi * 0.1 * t)
    
    y_final = y_vt + irregular
    
    # 4. ใส่สัญญาณรบกวน (Noise)
    y_final += 0.02 * np.random.randn(len(t))
    
    return y_final