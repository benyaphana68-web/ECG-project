import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Torsades de Pointes (TdP)
    ลักษณะ: Polymorphic VT ที่มียอดคลื่นบิดเกลียว (Twisting appearance)
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    # 1. ตั้งค่าความถี่ (เต้นเร็วมาก ~200 bpm)
    hr_tdp = 200
    rr_tdp = 60 / hr_tdp
    phase = (t % rr_tdp) / rr_tdp
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 2. สร้างโครงสร้าง QRS ที่กว้าง (Wide QRS)
    # TdP มักไม่มี P-wave ชัดเจน และมีลักษณะคล้ายคลื่น Sine ที่มีความแหลม
    y_qrs = (
        g(phase, 0.15, 0.04, 1.8) +  # ยอด R ที่กว้าง
        g(phase, 0.40, 0.06, -0.6)    # ยอด T สวนทาง
    )

    # 3. จุดสำคัญ: การจำลองการบิดเกลียว (The "Twisting" Effect)
    # ใช้ Sine wave ความถี่ต่ำ (0.2 Hz) มาคูณกับสัญญาณหลัก 
    # เพื่อให้ Amplitude สลับขั้วบวก/ลบ ช้าๆ เหมือนริบบิ้นบิด
    twisting_envelope = np.sin(2 * np.pi * 0.2 * t)
    y_tdp = y_qrs * twisting_envelope

    # 4. เพิ่ม Baseline drift และ Noise โกลาหลเล็กน้อย
    y_tdp += 0.15 * np.sin(2 * np.pi * 0.1 * t)
    y_tdp += 0.05 * np.random.randn(len(t))
    
    return y_tdp