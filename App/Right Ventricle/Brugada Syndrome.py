import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Brugada Syndrome (Type 1)
    ลักษณะ: J-point elevation, Coved ST elevation และ T-wave inversion
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. โครงสร้าง QRS พื้นฐาน (จำลองลักษณะ RBBB เบื้องต้น)
    # P-wave
    p_wave = g(phase, 0.18, 0.030, 0.12)
    # QRS ที่กว้างขึ้นเล็กน้อย
    qrs = (
        g(phase, 0.40, 0.015, -0.15) + # Q
        g(phase, 0.43, 0.012, 1.00) +  # R
        g(phase, 0.48, 0.025, -0.30)   # S ที่กว้างและลึก (ลักษณะ RBBB)
    )

    # 2. จุดสำคัญ: Brugada Type 1 Pattern (Coved appearance)
    # J-point ยกสูงขึ้นทันทีหลัง S-wave และลาดลง (Coved ST elevation)
    st_segment = g(phase, 0.55, 0.08, 0.50) 
    
    # 3. T-wave Inversion (หัวกลับ) - พบได้บ่อยใน Brugada Type 1
    t_wave = g(phase, 0.75, 0.06, -0.25)

    y_brs = p_wave + qrs + st_segment + t_wave

    # 4. ใส่ Baseline Wander และ Noise
    y_brs += 0.02 * np.sin(2 * np.pi * 0.33 * t)
    y_brs += 0.01 * np.random.randn(len(t))
    
    return y_brs