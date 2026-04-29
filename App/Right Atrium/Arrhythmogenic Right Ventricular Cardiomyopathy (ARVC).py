import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ ARVC 
    ลักษณะ: QRS กว้าง, Epsilon wave หลัง QRS, T-wave inversion และช่วง VT
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    np.random.seed(12)
    y_arvc = np.zeros(len(t))
    
    def g_time(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # 1. Beat ปกติของ ARVC
    rr_arvc = 0.82
    duration_total = t[-1] if len(t) > 0 else 30
    beats = np.arange(0, duration_total, rr_arvc)

    for bt in beats:
        # P wave
        y_arvc += g_time(t, bt + 0.16, 0.030, 0.10)
        # QRS prolongation (กว้างกว่าปกติ)
        y_arvc += g_time(t, bt + 0.38, 0.022, -0.18) # Q
        y_arvc += g_time(t, bt + 0.43, 0.026, 0.95)  # R
        y_arvc += g_time(t, bt + 0.49, 0.028, -0.45) # S
        
        # Epsilon wave (คลื่นจิ๋วหลัง QRS) - จุดเด่นของ ARVC 
        y_arvc += g_time(t, bt + 0.56, 0.012, 0.18)
        
        # T wave inversion
        y_arvc += g_time(t, bt + 0.74, 0.075, -0.35)

    # 2. ช่วงเกิด Ventricular Tachycardia (VT) (วินาทีที่ 12-17)
    vt_start, vt_end = 12, 17
    rr_vt = 0.32 # ~188 bpm
    vt_beats = np.arange(vt_start, vt_end, rr_vt)

    for bt in vt_beats:
        # ล้างสัญญาณปกติในช่วง VT (ทำให้ดูเหมือนเปลี่ยน Rhythm จริง)
        mask = (t >= bt) & (t < bt + rr_vt)
        y_arvc[mask] = 0 
        
        # ใส่คลื่น VT (QRS กว้างจากหัวใจห้องล่างขวา)
        y_arvc += g_time(t, bt + 0.06, 0.030, 0.35)
        y_arvc += g_time(t, bt + 0.11, 0.045, -1.20)
        y_arvc += g_time(t, bt + 0.17, 0.035, 0.55)
        y_arvc += g_time(t, bt + 0.27, 0.060, -0.25) # T-wave inversion

    # 3. ใส่สัญญาณรบกวน (Noise)
    y_arvc += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_arvc += 0.01 * np.random.randn(len(t))
    
    return y_arvc