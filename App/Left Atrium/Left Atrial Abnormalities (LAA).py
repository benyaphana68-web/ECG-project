import numpy as np

def generate_signal(t):
    """
    ฟังก์ชันสร้างสัญญาณ Left Atrial Abnormalities (LAA)
    t: รับค่าอาเรย์เวลาจากแอปหลัก เพื่อให้แอนิเมชันวิ่งตรงกัน
    """
    # กำหนดอัตราการเต้นของหัวใจให้ตรงกับ Normal (75 bpm) เพื่อการเปรียบเทียบที่ชัดเจน
    hr = 75
    rr = 60 / hr
    
    def g(phase, mu, sig, amp):
        return amp * np.exp(-0.5 * ((phase - mu) / sig) ** 2)

    # คำนวณเฟสของสัญญาณ
    phase = (t % rr) / rr
    
    # สร้างสัญญาณ LAA (เน้นลักษณะ P wave ที่กว้างหรือสูงผิดปกติ)
    # g(mu, sig, amp) -> mu: ตำแหน่ง, sig: ความกว้าง, amp: ความสูง
    y_disease = (
        g(phase, 0.18, 0.060, 0.25) +  # P wave: ปรับให้กว้างขึ้นและสูงขึ้น (LAA Characteristic)
        g(phase, 0.40, 0.012, -0.15) + # Q
        g(phase, 0.42, 0.008, 1.00) +  # R (ยอดแหลม QRS)
        g(phase, 0.44, 0.012, -0.25) + # S
        g(phase, 0.70, 0.060, 0.30)    # T
    )

    # เพิ่ม Baseline Wander และ Noise ตามโค้ดต้นฉบับของคุณ
    y_disease += 0.03 * np.sin(2 * np.pi * 0.33 * t)
    y_disease += 0.01 * np.random.randn(len(t))
    
    return y_disease

# ส่วนนี้ใช้ทดสอบรันไฟล์นี้เดี่ยวๆ (Standalone)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 250
    t_test = np.arange(0, 4, 1/fs)
    plt.plot(t_test, generate_signal(t_test), color='red')
    plt.title("LAA Signal Test")
    plt.grid(True)
    plt.show()