import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Polymorphic Ventricular Tachycardia (Torsades de Pointes)
    ลักษณะ: QRS กว้าง, อัตราเร็วมาก (200-250 bpm), 
    และมีการเปลี่ยนแปลงของ Amplitude เป็นคาบ (Twisting around baseline)
    """
    # 1. กำหนดความถี่หลักของ VT (ประมาณ 4 Hz = 240 bpm)
    vt_freq = 4.2 
    
    # 2. สร้างโครงสร้างคลื่นพื้นฐาน (Wide Complex) 
    # ใช้ Sine หลาย Harmonic เพื่อให้รูปคลื่นดูคล้าย QRS ที่กว้าง
    base_wave = (
        np.sin(2 * np.pi * vt_freq * t) +
        0.5 * np.sin(2 * np.pi * vt_freq * 2 * t + 0.8) +
        0.2 * np.sin(2 * np.pi * vt_freq * 3 * t + 0.4)
    )

    # 3. สร้างระบบการบิด (Twisting Envelope)
    # ใช้ Sine ความถี่ต่ำ (0.2 - 0.4 Hz) เพื่อทำให้ยอดคลื่นค่อยๆ สูงขึ้นและต่ำลง
    twisting_factor = np.sin(2 * np.pi * 0.3 * t)
    
    # คำนวณสัญญาณสุดท้าย
    # amplitude จะขยายและหดตาม twisting_factor
    y_poly = base_wave * twisting_factor * 1.5

    # 4. ทำ Smoothing ให้คลื่นดูมนแบบ Wide Complex (Ventricle origin)
    # ใช้ Moving average ขนาดเล็ก
    window_size = 5
    y_poly = np.convolve(y_poly, np.ones(window_size)/window_size, mode='same')

    # 5. ใส่ Baseline wander และ Noise
    y_poly += 0.05 * np.sin(2 * np.pi * 0.2 * t) # Low frequency drift
    y_poly += 0.015 * np.random.randn(len(t))    # High frequency noise
    
    return y_poly