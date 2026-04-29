import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Right Ventricular Hypertrophy (RVH)
    ลักษณะจำลอง:
    - R wave สูงและแหลม (Tall R-wave)
    - S wave ลึก (Deep S-wave)
    - T wave inversion (Right Ventricular Strain Pattern)
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. การสร้างคลื่น RVH
    # P-wave ปกติหรืออาจสูงเล็กน้อย
    p_wave = g(phase, 0.18, 0.030, 0.10)
    
    # QRS Complex: จุดชี้ขาดคือ R ที่สูงมากและ S ที่ลึก
    # ปรับความแหลม (sig) ให้เล็กลงเพื่อให้ดู "Sharp"
    qrs = (
        g(phase, 0.40, 0.012, -0.10) + # Q-wave เล็ก
        g(phase, 0.42, 0.006, 1.35)  + # Tall R-wave (แหลมและสูง)
        g(phase, 0.45, 0.015, -0.55)   # Deep S-wave
    )
    
    # 2. ST-T Segment: Right Ventricular Strain Pattern
    # มี ST depression และ T-wave Inversion ที่ชัดเจน
    st_t = (
        g(phase, 0.55, 0.030, -0.08) + # Mild ST depression
        g(phase, 0.72, 0.060, -0.35)    # Strong T-wave Inversion
    )

    y_rvh = p_wave + qrs + st_t

    # 3. ใส่สัญญาณรบกวนตามมาตรฐานแอป
    y_rvh += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_rvh += 0.01 * np.random.randn(len(t))
    
    return y_rvh