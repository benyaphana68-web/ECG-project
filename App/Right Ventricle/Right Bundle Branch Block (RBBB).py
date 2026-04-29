import numpy as np

def generate_signal(t):
    """
    สร้างสัญญาณ Right Bundle Branch Block (RBBB)
    ลักษณะเด่น: 
    1. QRS กว้าง (Wide QRS)
    2. rSR' complex (M-shape หรือหูกระต่าย)
    3. Terminal S-wave ที่กว้างและมน (Slurred S-wave)
    """
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    
    def g(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((p - mu) / sig) ** 2)

    # 1. โครงสร้างคลื่น RBBB
    # P-wave ปกติ
    p_wave = g(phase, 0.18, 0.030, 0.12)
    
    # QRS Complex แบบ rSR' (หัวใจสำคัญของ RBBB)
    qrs = (
        g(phase, 0.395, 0.010, -0.05) + # q เล็ก
        g(phase, 0.415, 0.009, 0.38)  + # r (ยอดแรกเล็ก)
        g(phase, 0.442, 0.014, -0.34) + # S (รอยหยักลง)
        g(phase, 0.490, 0.015, 1.02)  + # R' (ยอดที่สองสูงและกว้าง)
        g(phase, 0.530, 0.020, -0.15)   # Slurred S (terminal widening)
    )
    
    # 2. ST-T Segment
    # ใน RBBB มักพบ T-wave inversion หรือ Discordant T-wave ใน Lead ขวา
    st_t = (
        g(phase, 0.580, 0.030, -0.05) + # ST depression เล็กน้อย
        g(phase, 0.72, 0.060, -0.15)    # T-wave หัวกลับหรือต่ำลง
    )

    y_rbbb = p_wave + qrs + st_t

    # 3. ใส่สัญญาณรบกวนตามมาตรฐานแอป
    y_rbbb += 0.02 * np.sin(2 * np.pi * 0.33 * t)
    y_rbbb += 0.008 * np.random.randn(len(t))
    
    return y_rbbb