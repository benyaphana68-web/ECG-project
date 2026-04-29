import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Tricuspid Valve Disease (TVD)
    ลักษณะ: P-wave สูงแหลม (P pulmonale) จากภาวะ Right Atrial Enlargement (RAE) 
    และ T-wave inversion (ในบางรายที่มี Right Ventricular Strain)
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. สร้าง P-wave สูงแหลม (P pulmonale > 0.25 mV)
    # จุดเด่นหลักของ TVD ที่ส่งผลให้หัวใจห้องบนขวาโต
    p_wave = g(phase, 0.18, 0.025, 0.35) 

    # 2. สร้างโครงสร้าง QRS และ T-wave แบบ Inversion (กลับหัว)
    # จำลองภาวะ RV Strain ที่มักเกิดควบคู่กัน
    y_tvd = (
        p_wave +
        g(phase, 0.40, 0.012, -0.15) + # Q
        g(phase, 0.42, 0.008, 1.00) +  # R
        g(phase, 0.44, 0.012, -0.25) + # S
        g(phase, 0.70, 0.060, -0.25)   # T-wave Inversion (ปรับ amp เป็นลบ)
    )

    # 3. ใส่สัญญาณรบกวน (Noise) และ Baseline wander
    y_tvd += 0.02 * np.sin(2 * np.pi * 0.33 * t)
    y_tvd += 0.01 * np.random.randn(len(t))
    
    return y_tvd