import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Persistent Left Superior Vena Cava (PLSVC)
    ลักษณะจำลอง: P-wave ขยายกว้างและสูงขึ้น (มักสัมพันธ์กับ Coronary Sinus ที่ขยายตัว)
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. สร้าง P-wave ที่ผิดปกติ (ขยายกว้างและสูงตาม Logic ของคุณ)
    # PLSVC มักทำให้ Coronary Sinus โต ซึ่งส่งผลต่อรูปร่าง P-wave ใน ECG
    p_wave = g(phase, 0.18, 0.060, 0.30)

    # 2. สร้างโครงสร้าง QRS และ T wave
    # ปรับแต่งเล็กน้อยให้สอดคล้องกับลักษณะหัวใจที่มีโครงสร้างหลอดเลือดผิดปกติ
    y_plsvc = (
        p_wave +
        g(phase, 0.40, 0.020, -0.15) + # Q
        g(phase, 0.42, 0.010, 1.05) +  # R (สูงขึ้นเล็กน้อยตามที่คุณตั้งค่า)
        g(phase, 0.44, 0.020, -0.30) + # S
        g(phase, 0.70, 0.060, 0.30)    # T
    )

    # 3. ใส่สัญญาณรบกวน (Noise) และ Baseline wander
    y_plsvc += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_plsvc += 0.01 * np.random.randn(len(t))
    
    return y_plsvc