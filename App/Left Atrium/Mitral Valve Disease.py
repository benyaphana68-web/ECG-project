import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Mitral Valve Disease (P mitrale)
    t: รับค่าอาเรย์เวลาจากแอปหลัก เพื่อให้แอนิเมชันวิ่งตรงกัน
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr

    def g(phase_input, mu, sig, amp):
        return amp * np.exp(-0.5 * ((phase_input - mu) / sig) ** 2)

    # ลักษณะจำลอง Mitral Valve Disease: P wave สองยอด (P mitrale)
    y_disease = (
        g(phase, 0.145, 0.030, 0.12) +   # P peak แรก
        g(phase, 0.215, 0.030, 0.14) +   # P peak สอง (M-shape)

        g(phase, 0.40, 0.012, -0.15) +   # Q
        g(phase, 0.42, 0.008, 1.00) +    # R (ยอดแหลมต้องสูงชัดเจน)
        g(phase, 0.44, 0.012, -0.25) +   # S

        g(phase, 0.70, 0.060, 0.28)      # T
    )

    # เพิ่ม Baseline wander และ Noise เล็กน้อยเพื่อให้สมจริง
    y_disease += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_disease += 0.01 * np.random.randn(len(t))
    
    return y_disease

# ส่วนสำหรับรันทดสอบแยกไฟล์ (Standalone Test)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 250
    t_test = np.arange(0, 4, 1/fs)
    plt.figure(figsize=(10, 4))
    plt.plot(t_test, generate_signal(t_test), color='red')
    plt.title("Mitral Valve Disease (P mitrale) Test Plot")
    plt.grid(True)
    plt.show()