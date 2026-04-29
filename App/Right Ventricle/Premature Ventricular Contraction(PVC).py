import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Premature Ventricular Contraction (PVC)
    ลักษณะ: จังหวะปกติที่มี PVC แทรกมาก่อนเวลา (Early beat) 
    PVC จะมี QRS กว้าง, ไม่มี P-wave และตามด้วยช่วงพัก (Compensatory Pause)
    """
    np.random.seed(42)
    y_pvc = np.zeros(len(t))
    duration = t[-1] if len(t) > 0 else 30
    fs = 250
    hr = 75
    rr = 60 / hr

    def gaussian(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    beat_time = 0.2
    beat_idx = 0
    # กำหนดตำแหน่งที่จะเกิด PVC (สุ่มหรือกำหนดตายตัว)
    pvc_beats = {3, 8, 14, 20, 26} 

    while beat_time < duration:
        if beat_idx in pvc_beats:
            # --- PVC Beat (Early & Wide) ---
            # เกิดก่อนเวลาปกติ (ประมาณ 60% ของ RR interval)
            bt = beat_time - (rr * 0.35)
            
            # รูปร่าง PVC: กว้างและสูง/ลึกผิดปกติ (Bizarre QRS)
            y_pvc += gaussian(t, bt + 0.10, 0.045, -0.45) # Q กว้าง
            y_pvc += gaussian(t, bt + 0.16, 0.050, 1.60)  # R สูงและกว้าง
            y_pvc += gaussian(t, bt + 0.22, 0.045, -0.65) # S กว้าง
            y_pvc += gaussian(t, bt + 0.40, 0.080, -0.50) # T-wave กลับทิศ (Discordant)

            # Compensatory Pause: หัวใจจะพักนานขึ้นเพื่อให้จังหวะถัดไปกลับมาตรง Loop เดิม
            beat_time += rr * 2.0 
        else:
            # --- Normal Beat ---
            bt = beat_time
            y_pvc += gaussian(t, bt + 0.00, 0.025, 0.12)  # P
            y_pvc += gaussian(t, bt + 0.14, 0.010, -0.15) # Q
            y_pvc += gaussian(t, bt + 0.16, 0.007, 1.00)  # R
            y_pvc += gaussian(t, bt + 0.18, 0.010, -0.25) # S
            y_pvc += gaussian(t, bt + 0.32, 0.050, 0.30)  # T
            
            beat_time += rr

        beat_idx += 1

    # ใส่สัญญาณรบกวนตามมาตรฐานแอป
    y_pvc += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_pvc += 0.01 * np.random.randn(len(t))
    
    return y_pvc