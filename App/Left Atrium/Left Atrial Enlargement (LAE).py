import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Left Atrial Enlargement (LAE)
    ลักษณะสำคัญ: P wave กว้างกว่าปกติ (> 0.12s) และอาจมี 2 ยอด (M-shape)
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(phase_input, mu, sig, amp):
        return amp * np.exp(-0.5 * ((phase_input - mu) / sig) ** 2)

    # ปรับลักษณะสัญญาณ LAE (P wave กว้างขึ้น)
    y_disease = (
        # P wave: ปรับความกว้าง (sig) ให้มากขึ้นจาก 0.03 เป็น 0.05
        g(phase, 0.18, 0.050, 0.20) + 
        g(phase, 0.40, 0.012, -0.15) + # Q
        g(phase, 0.42, 0.008, 1.00) +  # R (ยอดแหลมต้องสูงชัดเจน)
        g(phase, 0.44, 0.012, -0.25) + # S
        g(phase, 0.70, 0.060, 0.30)    # T
    )

    # เพิ่ม Baseline wander และ Noise ตามสูตรเดิมของคุณ
    y_disease += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_disease += 0.01 * np.random.randn(len(t))
    
    return y_disease