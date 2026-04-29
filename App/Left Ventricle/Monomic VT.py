import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Monomorphic VT โดยคงค่าพารามิเตอร์เดิมของผู้ใช้
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    def g_func(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # ใช้ตัวเลขเดิมเป๊ะๆ
    y_mvt = np.zeros(len(t))
    rr_mvt = 0.40  # ~150 bpm
    duration = t[-1] - t[0] if len(t) > 0 else 30
    
    # วนลูปสร้าง Beat ตามระยะเวลา t ที่ได้รับมา
    # ปรับ Logic เล็กน้อยเพื่อให้ Beat วิ่งตามเวลาจริงของแอป
    t_start = t[0] - (t[0] % rr_mvt)
    for bt in np.arange(t_start, t[-1] + rr_mvt, rr_mvt):
        # QRS กว้าง (ใช้ตัวเลขเดิมเป๊ะๆ)
        y_mvt += g_func(t, bt + 0.10, 0.015, -0.30)   # Q กว้าง
        y_mvt += g_func(t, bt + 0.13, 0.025,  1.60)   # R สูง + กว้าง
        y_mvt += g_func(t, bt + 0.17, 0.020, -0.50)   # S กว้าง
        # T wave สวนทาง (ใช้ตัวเลขเดิมเป๊ะๆ)
        y_mvt += g_func(t, bt + 0.28, 0.055, -0.45)   # T กลับทิศ

    # เพิ่ม Noise และ Baseline Wander (ใช้ตัวเลขเดิมเป๊ะๆ)
    y_mvt += 0.02 * np.sin(2 * np.pi * 0.3 * t)
    y_mvt += 0.015 * np.random.randn(len(t))
    
    return y_mvt