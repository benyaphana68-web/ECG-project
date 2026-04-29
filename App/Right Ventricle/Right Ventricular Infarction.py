import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Right Ventricular Infarction (RVI)
    ลักษณะจำลอง:
    - Q-wave ลึกขึ้นเล็กน้อย (Pathological Q-wave)
    - R-wave ลดความสูงลง (Loss of R-wave amplitude)
    - ST-segment elevation ที่ชัดเจนและต่อเนื่องจากจุด J-point
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. การสร้างโครงสร้าง QRS ที่ผิดปกติ
    p_wave = g(phase, 0.18, 0.030, 0.10)   # P-wave ปกติ
    qrs = (
        g(phase, 0.38, 0.012, -0.25) +    # Q-wave ลึก (Infarction sign)
        g(phase, 0.42, 0.010, 0.70)  +    # R-wave ต่ำลง (Reduced voltage)
        g(phase, 0.45, 0.012, -0.20)      # S-wave
    )

    # 2. จุดสำคัญ: ST Elevation (Plateau Shape)
    # ใช้ฟังก์ชัน Sigmoid สองตัวหักล้างกันเพื่อสร้างลักษณะ ST ที่ยกสูงค้างไว้
    st_start = 0.46
    st_end = 0.65
    st_height = 0.35 # ยกสูง 0.35 mV
    st_elevation = st_height * (
        1 / (1 + np.exp(-(phase - st_start) / 0.01)) - 
        1 / (1 + np.exp(-(phase - st_end) / 0.03))
    )
    
    # 3. T-wave ที่อาจจะแบนลงหรือผิดปกติ
    t_wave = g(phase, 0.72, 0.070, 0.12)

    y_rvi = p_wave + qrs + st_elevation + t_wave

    # 4. ใส่ Baseline wander และ Noise ตามมาตรฐาน
    y_rvi += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_rvi += 0.01 * np.random.randn(len(t))
    
    return y_rvi