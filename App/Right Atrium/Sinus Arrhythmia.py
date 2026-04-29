import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Sinus Arrhythmia
    ลักษณะ: รูปร่างคลื่นปกติ (P-QRS-T) แต่จังหวะการเต้นเปลี่ยนตามการหายใจ 
    (เร็วขึ้นเมื่อหายใจเข้า, ช้าลงเมื่อหายใจออก)
    """
    np.random.seed(42)
    y_sa = np.zeros(len(t))
    duration = t[-1] if len(t) > 0 else 30
    
    def gaussian(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # 1. จำลองตำแหน่ง Beat (bt) ที่มีความแปรปรวนตาม Respiratory Cycle
    bt = 0
    while bt < duration:
        # RR interval พื้นฐาน 0.8s (75 bpm) 
        # แกว่งขึ้นลงตาม Sine wave ความถี่ 0.25 Hz (หายใจ 1 รอบทุก 4 วินาที)
        rr_base = 0.80
        rr_vary = 0.18 # ความกว้างของการแกว่งจังหวะ
        rr_current = rr_base + rr_vary * np.sin(2 * np.pi * 0.25 * bt)

        # 2. ใส่คลื่นปกติลงในตำแหน่ง bt ที่คำนวณได้
        # (ไม่มี P-wave หาย หรือ QRS ผิดปกติ)
        y_sa += gaussian(t, bt + 0.10, 0.025, 0.12) # P
        y_sa += gaussian(t, bt + 0.28, 0.010, -0.15) # Q
        y_sa += gaussian(t, bt + 0.30, 0.007, 1.00)  # R
        y_sa += gaussian(t, bt + 0.32, 0.010, -0.25) # S
        y_sa += gaussian(t, bt + 0.48, 0.050, 0.30)  # T

        bt += rr_current

    # 3. ใส่ Baseline wander และ Noise
    y_sa += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_sa += 0.01 * np.random.randn(len(t))