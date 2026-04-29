import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Monomorphic Ventricular Tachycardia (Monomorphic VT)
    ลักษณะ: อัตราการเต้นเร็วสม่ำเสมอ (150-200 bpm), QRS กว้าง (Wide Complex), 
    รูปร่างเหมือนกันทุก Beat และไม่มี P-wave
    """
    y_mvt = np.zeros(len(t))
    duration = t[-1] if len(t) > 0 else 30
    
    def g_time(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # 1. ตั้งค่าจังหวะ Monomorphic VT
    # rr_vt = 0.33 (~182 bpm) สม่ำเสมอมาก
    rr_vt = 0.33
    beats = np.arange(0, duration, rr_vt)

    for bt in beats:
        # Monomorphic VT รูปร่าง QRS จะกว้างและมีลักษณะมน (Slurred) 
        # เนื่องจากจุดกำเนิดกระแสไฟมาจากกล้ามเนื้อหัวใจห้องล่างโดยตรง
        y_mvt += g_time(t, bt + 0.08, 0.030, -0.35) # Q กว้าง
        y_mvt += g_time(t, bt + 0.14, 0.045, 1.35)  # R กว้างและมน
        y_mvt += g_time(t, bt + 0.22, 0.040, -0.75) # S กว้าง
        y_mvt += g_time(t, bt + 0.32, 0.070, -0.30) # T-wave ผิดทิศทาง (Discordant)

    # 2. ใส่สัญญาณรบกวน (Noise) และ Baseline wander
    y_mvt += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_mvt += 0.01 * np.random.randn(len(t))
    
    return y_mvt