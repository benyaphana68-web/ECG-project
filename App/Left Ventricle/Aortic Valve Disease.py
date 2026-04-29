import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Aortic Valve Disease
    ลักษณะ: R wave สูง (LVH), ST depression และ T wave inversion (LV Strain)
    t: รับค่าอาเรย์เวลาจากแอปหลัก
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr

    def g(phase_input, mu, sig, amp):
        return amp * np.exp(-0.5 * ((phase_input - mu) / sig) ** 2)

    # คำนวณ ST depression แบบนุ่มนวล
    st_start = 0.47
    st_end = 0.64
    st_depression = -0.24 * (
        1 / (1 + np.exp(-(phase - st_start) / 0.010))
        - 1 / (1 + np.exp(-(phase - st_end) / 0.010))
    )

    # สร้างรูปคลื่น Aortic Valve Disease
    y_disease = (
        g(phase, 0.18, 0.030, 0.10) +    # P wave
        g(phase, 0.40, 0.012, -0.18) +   # Q wave
        g(phase, 0.42, 0.009, 1.35) +    # R สูงชัดเจน (LVH pattern)
        g(phase, 0.445, 0.014, -0.35) +  # S ลึกขึ้น
        st_depression +                  # ST depression
        g(phase, 0.72, 0.075, -0.40)     # T wave inversion (Strain)
    )

    # เพิ่มสัญญาณรบกวนเล็กน้อยให้ดูสมจริง
    y_disease += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_disease += 0.01 * np.random.randn(len(t))
    
    return y_disease

# ส่วนสำหรับทดสอบรันแยกไฟล์ (Standalone Test)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 250
    t_test = np.arange(0, 4, 1/fs)
    plt.figure(figsize=(10, 5))
    # ใช้พื้นหลังดำเหมือนจอมอนิเตอร์จริงเพื่อเช็คสี
    plt.gca().set_facecolor('black')
    plt.plot(t_test, generate_signal(t_test), color='#ec7063', lw=1.5)
    plt.title("Aortic Valve Disease Test Plot", color='white')
    plt.ylim(-1.0, 1.8)
    plt.grid(color='#333333', linestyle=':')
    plt.show()