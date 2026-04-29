import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Pulmonary Embolism (PE)
    ลักษณะเด่น: 
    1. Sinus Tachycardia (เต้นเร็ว > 100 bpm)
    2. Right Heart Strain Pattern (S-wave ลึก, T-wave Inversion)
    3. เลียนแบบลักษณะ S1Q3T3 (ใน Lead ที่แสดงผล)
    """
    # 1. ตั้งค่าความเร็ว (Tachycardia ~122 bpm)
    hr_pe = 122
    rr_pe = 60 / hr_pe
    phase = (t % rr_pe) / rr_pe
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 2. สร้างโครงสร้างคลื่นที่แสดงลักษณะความผิดปกติ
    # P-wave เล็ก (เนื่องจากเต้นเร็ว)
    p_wave = g(phase, 0.145, 0.020, 0.07)
    
    # QRS Complex: S ลึก (ลักษณะ Right Heart Strain / S1) 
    # และ Q-wave ที่เห็นชัดขึ้น (Q3)
    qrs = (
        g(phase, 0.380, 0.008, -0.16) + # Q-wave (ลึกกว่าปกติเล็กน้อย)
        g(phase, 0.405, 0.007, 0.92)  + # R-wave
        g(phase, 0.435, 0.015, -0.45)   # S-wave ลึก (เด่นชัด)
    )
    
    # 3. ST-T Segment: T-wave Inversion (หัวกลับ)
    # แสดงถึงภาวะหัวใจห้องขวาทำงานหนัก (Right Ventricular Strain)
    st_t = (
        g(phase, 0.55, 0.030, -0.06) + # ST Depression เล็กน้อย
        g(phase, 0.68, 0.050, -0.30)   # T-wave Inversion (หัวกลับชัดเจน)
    )

    y_pe = p_wave + qrs + st_t

    # 4. ใส่ Baseline wander (เน้นความถี่ที่สอดคล้องกับการหายใจเร็ว) และ Noise
    y_pe += 0.02 * np.sin(2 * np.pi * 0.45 * t)
    y_pe += 0.008 * np.random.randn(len(t))
    
    return y_pe