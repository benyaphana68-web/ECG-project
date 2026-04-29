import numpy as np
import random

def generate_signal(t):
    """
    สร้างสัญญาณ Atrial Fibrillation (AF)
    ลักษณะ: ไม่มี P wave (แทนด้วย f-waves) และ RR interval ไม่สม่ำเสมอ
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    np.random.seed(42)
    random.seed(42)
    y_af = np.zeros(len(t))
    fs = 250
    duration = t[-1] if len(t) > 0 else 30

    def gaussian(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # 1. สร้าง Fibrillatory baseline (f-waves) แทน P wave
    # ใช้การผสมคลื่น Sine หลายความถี่ในช่วง 4-9 Hz เพื่อความสมจริง
    for freq in np.arange(4.0, 9.0, 0.7):
        amp_f = np.random.uniform(0.02, 0.06)
        phase_f = np.random.uniform(0, 2 * np.pi)
        y_af += amp_f * np.sin(2 * np.pi * freq * t + phase_f)

    # 2. สร้าง QRS ปกติ แต่ RR interval ไม่สม่ำเสมอ
    bt = 0
    while bt < duration:
        # สุ่มช่วง RR (จังหวะสุ่ม 60-120 bpm)
        RR = random.uniform(0.50, 1.10)
        bt += RR
        
        if bt < duration:
            # QRS complex (คงรูปร่างเดิมของคุณไว้)
            y_af += gaussian(t, bt + 0.07, 0.010, -0.15) # Q
            y_af += gaussian(t, bt + 0.09, 0.007, 1.00)  # R
            y_af += gaussian(t, bt + 0.11, 0.010, -0.25) # S
            y_af += gaussian(t, bt + 0.27, 0.050, 0.28)  # T

    # 3. ใส่ Noise เล็กน้อย
    y_af += 0.01 * np.random.randn(len(t))
    
    return y_af