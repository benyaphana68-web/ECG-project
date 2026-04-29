import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Left_Ventricular_Failure โดยคงค่าตัวเลขเดิมของผู้ใช้
    t: รับค่าอาเรย์เวลาที่ส่งมาจากแอปหลัก
    """
    # คงค่าพื้นฐานเดิม
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr

    def g(mu, sig, amp):
        return amp * np.exp(-0.5*((phase - mu)/sig)**2)

    # 1. คำนวณ ST depression แบบ plateau (ใช้ตัวเลขเดิมเป๊ะๆ)
    st_start = 0.46
    st_end = 0.62
    st_depression = -0.22 * (
        1 / (1 + np.exp(-(phase - st_start) / 0.008))
        - 1 / (1 + np.exp(-(phase - st_end) / 0.008))
    )

    # 2. สร้าง Waveform (ใช้ตัวเลขเดิมเป๊ะๆ)
    y_lvf = (
        g(0.18, 0.030,  0.10) +   # P wave
        g(0.40, 0.012, -0.18) +   # Q wave
        g(0.42, 0.009,  1.35) +   # R wave สูงขึ้น
        g(0.44, 0.012, -0.30) +   # S wave
        st_depression +           # ST depression
        g(0.72, 0.070, -0.35)     # T wave inversion
    )

    # 3. ใส่ Noise และ Baseline Wander (ใช้ตัวเลขเดิมเป๊ะๆ)
    y_lvf += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_lvf += 0.01 * np.random.randn(len(t))
    
    return y_lvf # ส่งค่ากลับไปพล็อตที่แอปหลัก