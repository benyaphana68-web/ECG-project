import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Atrial Flutter (AFL)
    ลักษณะ: P-wave ถูกแทนที่ด้วย F-waves ลักษณะฟันปลา (Sawtooth) 
    ที่มีความถี่สม่ำเสมอ (ประมาณ 250-350 bpm)
    """
    y_afl = np.zeros(len(t))
    
    # 1. สร้าง Flutter Waves (Sawtooth Pattern)
    # ความถี่ atrial rate ประมาณ 300 bpm (5 Hz)
    f_freq = 5.0 
    # สร้างฟันปลาโดยใช้ฟังก์ชัน Linear Sawtooth เพื่อความคมชัด
    # ปรับให้ Amplitude อยู่ที่ประมาณ 0.2-0.3 mV
    flutter = 0.25 * (np.abs((t * f_freq) % 1.0 - 0.5) * 2 - 0.5)
    y_afl += flutter

    # 2. สร้าง QRS Complex (จังหวะ 4:1 block)
    # คือทุกๆ 4 flutter waves จะมี QRS 1 ครั้ง (Ventricular rate ~75 bpm)
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # ใส่ QRS และ T-wave ปกติลงไปทับบน flutter waves
    y_qrs = (
        g(phase, 0.40, 0.012, -0.15) + # Q
        g(phase, 0.42, 0.008, 1.00) +  # R
        g(phase, 0.44, 0.012, -0.25) + # S
        g(phase, 0.70, 0.060, 0.25)    # T
    )
    
    y_afl += y_qrs

    # 3. ใส่สัญญาณรบกวน (Noise) ตามที่คุณตั้งค่าไว้
    y_afl += 0.01 * np.random.randn(len(t))
    
    return y_afl