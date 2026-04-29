import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Hypertensive Heart Disease
    ลักษณะ: R wave สูง (LVH), ST depression เล็กน้อย และ T wave inversion (Strain pattern)
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr

    def g(phase_input, mu, sig, amp):
        return amp * np.exp(-0.5 * ((phase_input - mu) / sig) ** 2)

    # 1. คำนวณ ST depression (ใช้ Logistic function เพื่อความสมูท)
    st_start = 0.48
    st_end = 0.62
    st_depression = -0.15 * (
        1 / (1 + np.exp(-(phase - st_start) / 0.01)) - 
        1 / (1 + np.exp(-(phase - st_end) / 0.01))
    )

    # 2. สร้างรูปคลื่นหัวใจ
    y_disease = (
        g(phase, 0.18, 0.030, 0.11) +   # P wave
        g(phase, 0.40, 0.012, -0.15) +  # Q wave
        g(phase, 0.42, 0.009, 1.25) +   # R สูงชัดเจน (บ่งบอก LVH)
        g(phase, 0.44, 0.012, -0.25) +  # S wave
        st_depression +                 # เพิ่มอาการ ST depression (Strain)
        g(phase, 0.72, 0.070, -0.25)    # T wave inversion (Strain pattern)
    )

    # 3. ใส่สัญญาณรบกวนเล็กน้อยให้ดูสมจริง (Baseline wander + Random noise)
    y_disease += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_disease += 0.01 * np.random.randn(len(t))
    
    return y_disease

# ส่วนสำหรับทดสอบรันแยกไฟล์ (Standalone Test)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 250
    t_test = np.arange(0, 4, 1/fs)
    plt.figure(figsize=(10, 5))
    plt.gca().set_facecolor('black') # จำลองพื้นหลังดำ
    plt.plot(t_test, generate_signal(t_test), color='red', lw=1.5)
    plt.title("Hypertensive Heart Disease (LVH + Strain) Test", color='white')
    plt.grid(color='#333333', linestyle=':')
    plt.ylim(-0.8, 1.6)
    plt.show()