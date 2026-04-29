import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Heart Failure (HF) 
    ลักษณะ: QRS Complex กว้างขึ้น (Widened QRS) และ T-wave หัวกลับ (Inversion)
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(phase_input, mu, sig, amp):
        return amp * np.exp(-0.5 * ((phase_input - mu) / sig) ** 2)

    # ปรับลักษณะ Heart Failure
    y_disease = (
        g(phase, 0.18, 0.030, 0.12) +   # P wave
        g(phase, 0.38, 0.025, -0.15) +  # Q wave (เริ่มเร็วขึ้นและกว้างขึ้น)
        g(phase, 0.42, 0.015, 1.00) +   # R wave (กว้างขึ้นเล็กน้อย)
        g(phase, 0.46, 0.025, -0.25) +  # S wave (กว้างขึ้น)
        g(phase, 0.70, 0.080, -0.35)    # T-wave inversion (ปรับ amp เป็นลบ และ sig กว้างขึ้น)
    )

    # เพิ่ม Baseline wander และ Noise เล็กน้อย
    y_disease += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_disease += 0.01 * np.random.randn(len(t))
    
    return y_disease

# ส่วนสำหรับทดสอบรันแยกไฟล์
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 250
    t_test = np.arange(0, 4, 1/fs)
    plt.figure(figsize=(10, 4))
    plt.gca().set_facecolor('black')
    plt.plot(t_test, generate_signal(t_test), color='tomato', lw=1.5)
    plt.title("Heart Failure (HF) Signal Test", color='white')
    plt.grid(color='#333333', linestyle=':')
    plt.show()