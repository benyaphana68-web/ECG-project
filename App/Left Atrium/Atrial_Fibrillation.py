import numpy as np
import random

# แก้ไขให้ฟังก์ชันรับค่า t (ตัวแปรเวลา) จากไฟล์หลัก
def generate_signal(t): 
    """ฟังก์ชันสำหรับสร้างสัญญาณ Atrial Fibrillation (AF)"""
    duration = t[-1]
    y_af = np.zeros(len(t))
    
    # 1. สร้าง Baseline ที่สั่นไหว (Fibrillatory waves)
    for freq in np.arange(4.0, 9.0, 0.7):
        amp_f = np.random.uniform(0.02, 0.06)
        phase_f = np.random.uniform(0, 2 * np.pi)
        y_af += amp_f * np.sin(2 * np.pi * freq * t + phase_f)

    # ฟังก์ชันเสริมสำหรับสร้างคลื่น QRS
    def gaussian(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # 2. สร้างจังหวะการเต้นที่ผิดปกติ (Irregular RR interval)
    bt = 0
    while bt < duration:
        # สุ่มช่วงเวลาการเต้น (RR interval)
        RR = random.uniform(0.50, 1.10) 
        bt += RR
        
        if bt < duration:
            # สร้างคลื่น QRS-T (AF มักไม่มีคลื่น P)
            y_af += gaussian(t, bt + 0.07, 0.010, -0.15)  # Q
            y_af += gaussian(t, bt + 0.09, 0.007,  1.00)  # R
            y_af += gaussian(t, bt + 0.11, 0.010, -0.25)  # S
            y_af += gaussian(t, bt + 0.27, 0.050,  0.28)  # T
            
    # เติม Noise เล็กน้อยให้สมจริงเหมือนในวิดีโอตัวอย่าง
    y_af += 0.01 * np.random.randn(len(t))
    
    # --- จุดที่สำคัญที่สุด ---
    return y_af  # ต้องมีการ return ค่าสัญญาณกลับไปเสมอ