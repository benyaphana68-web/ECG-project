import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Polymorphic VT (PVT) 
    ลักษณะ: QRS รูปร่างเปลี่ยนทุก beat (แกนหมุน) และสลับความสูง
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    # ตั้งค่าพารามิเตอร์ตาม Logic เดิมของคุณเป๊ะๆ
    np.random.seed(7)
    y_pvt = np.zeros(len(t))
    rr_pvt = 0.33  # ~180 bpm
    
    def g_func(tt, center, sig, amp):
        return amp * np.exp(-0.5 * ((tt - center) / sig) ** 2)

    # คำนวณช่วง beats ทั้งหมดในช่วงเวลา t
    # เริ่มต้นจาก beat แรกที่อาจอยู่นอกขอบเขต t เล็กน้อยเพื่อให้กราฟต่อเนื่อง
    duration_total = 30 # สมมติระยะเวลาสูงสุด
    beats = np.arange(0, duration_total + 1, rr_pvt)

    for idx, bt in enumerate(beats):
        # Logic การเปลี่ยนรูปร่างคลื่น (ใช้ตัวเลขเดิมของคุณ)
        amp_r   = np.random.uniform(0.6, 1.8)
        sign    = 1 if np.sin(2 * np.pi * idx / 7) >= 0 else -1
        sig_qrs = np.random.uniform(0.025, 0.050)
        jitter  = np.random.uniform(-0.02, 0.02)

        # สร้างคลื่นเฉพาะในช่วงที่ t กำลังแสดงผลเพื่อประหยัด RAM
        y_pvt += g_func(t, bt + 0.10 + jitter, sig_qrs * 0.6, -sign * amp_r * 0.25) # Q
        y_pvt += g_func(t, bt + 0.13 + jitter, sig_qrs,         sign * amp_r)        # R
        y_pvt += g_func(t, bt + 0.17 + jitter, sig_qrs * 0.7, -sign * amp_r * 0.35) # S
        y_pvt += g_func(t, bt + 0.26 + jitter, sig_qrs * 1.5, -sign * amp_r * 0.25) # T

    # เพิ่ม Baseline wander ช้าๆ (Torsades-like) และ Noise
    y_pvt += 0.15 * np.sin(2 * np.pi * 0.18 * t)
    y_pvt += 0.02 * np.random.randn(len(t))
    
    return y_pvt