import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Myocardial Infarction (MI)
    ลักษณะ: ST-segment deviation และ T-wave inversion
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr

    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. สร้างโครงสร้าง ECG พื้นฐาน (P, Q, R, S)
    # ใช้ค่า Amplitude เดิมของคุณ
    y_base = (
        g(phase, 0.18, 0.030, 0.12) +  # P wave
        g(phase, 0.40, 0.012, -0.15) + # Q wave
        g(phase, 0.42, 0.008, 1.00) +  # R wave
        g(phase, 0.44, 0.012, -0.25)   # S wave
    )

    # 2. จำลอง ST Elevation/Depression และ T-wave Inversion ให้คงที่
    # แทนที่จะใช้ sin(tt) ซึ่งจะทำให้กราฟแกว่งขึ้นลงไม่หยุด 
    # เราจะใช้การปรับที่ Phase เพื่อให้ทุก Beat มีลักษณะเหมือนกัน
    
    # จำลอง ST Segment ที่ยกตัวหรือกดลง (ตามตำแหน่ง phase 0.45 - 0.65)
    st_segment = -0.20 * (1 / (1 + np.exp(-(phase - 0.46) / 0.01)) - 1 / (1 + np.exp(-(phase - 0.65) / 0.01)))
    
    # T-wave inversion (กลับข้างจากปกติ)
    t_wave = g(phase, 0.70, 0.060, -0.30) 

    y_mi = y_base + st_segment + t_wave

    # 3. ใส่สัญญาณรบกวน (Noise) ตามที่คุณตั้งค่าไว้
    y_mi += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_mi += 0.01 * np.random.randn(len(t))
    
    return y_mi