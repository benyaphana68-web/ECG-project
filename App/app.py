import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import time
import inspect

# --- 1. ตั้งค่า Theme ของหน้าเว็บ ---
st.set_page_config(page_title="Professional ECG Monitor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stRadio > div { background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
    h1, h2, h3, p, span, label { color: white !important; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white !important; border-radius: 10px; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 2. ฟังก์ชันจัดการสัญญาณ ---
def generate_normal_signal(t):
    """สร้างสัญญาณ ECG ปกติ (75 bpm)"""
    hr = 75
    rr = 60 / hr
    phase = (t % rr) / rr
    def g(mu, sig, amp): return amp * np.exp(-0.5*((phase - mu)/sig)**2)
    y = (g(0.18, 0.030, 0.12) + g(0.40, 0.012, -0.15) + 
         g(0.42, 0.008, 1.00) + g(0.44, 0.012, -0.25) + g(0.70, 0.060, 0.30))
    return y + 0.01 * np.random.randn(len(t))

def get_disease_signal_from_file(folder, file_name, t):
    """ดึงข้อมูลจากไฟล์โรคย่อยแบบ Dynamic"""
    file_path = os.path.join(BASE_DIR, folder, f"{file_name}.py")
    if not os.path.exists(file_path): return np.zeros(len(t))
    try:
        spec = importlib.util.spec_from_file_location("mod", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # ตรวจหาฟังก์ชันในไฟล์ย่อย
        for func_name in ['generate_signal', 'afl_waveform', 'generate_af', 'left_ventricular_failure_waveform']:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                sig = inspect.signature(func)
                return func(t) if len(sig.parameters) >= 1 else func()
    except Exception as e:
        st.error(f"Error in {file_name}.py: {e}")
    return np.zeros(len(t))

# --- 3. หน้าจอมอนิเตอร์ Pop-up (จุดที่แก้ไขสเกล 2.0 mV) ---
@st.dialog("Real-time ECG Monitoring Comparison", width="large")
def show_comparison_monitor(folder, disease):
    st.markdown(f"📊 **วิเคราะห์เปรียบเทียบ:** `{disease}`")
    plot_placeholder = st.empty()
    
    fs = 250
    t_full = np.arange(0, 20, 1/fs)
    y_normal = generate_normal_signal(t_full)
    y_disease = get_disease_signal_from_file(folder, disease, t_full)
    
    win_sec = 4.0
    win_size = int(win_sec * fs)
    
    # วนลูปสร้าง Animation
    for i in range(0, len(t_full) - win_size, 12):
        t_window = t_full[i:i+win_size] - t_full[i]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.patch.set_facecolor('#0e1117') 
        
        # กราฟบน (Normal)
        ax1.plot(t_window, y_normal[i:i+win_size], color='#5dade2', lw=1.5)
        ax1.set_title("NORMAL ECG (75 BPM)", color='#5dade2', loc='left', fontsize=11, fontweight='bold')
        
        # กราฟล่าง (Disease)
        ax2.plot(t_window, y_disease[i:i+win_size], color='#ec7063', lw=1.5)
        ax2.set_title(f"DISEASE: {disease.upper()}", color='#ec7063', loc='left', fontsize=11, fontweight='bold')
        
        for idx, ax in enumerate([ax1, ax2]):
            ax.set_facecolor('#000000') # พื้นหลังดำสนิท
            
            # --- ปรับสเกลแกน Y ให้กราฟล่างถึง 2.0 ---
            if idx == 0:
                ax.set_ylim(-0.5, 1.4) 
            else:
                ax.set_ylim(-0.8, 2.0) # แก้ไขจาก 1.4 เป็น 2.0 ตามคำขอ
            
            ax.set_xlim(0, win_sec)
            
            # เส้น Grid แบบ ECG Monitor
            ax.grid(which='major', color='#333333', linestyle='-', linewidth=0.8)
            ax.grid(which='minor', color='#151515', linestyle=':', linewidth=0.5)
            ax.minorticks_on()
            
            ax.tick_params(colors='white', labelsize=8)
            ax.set_ylabel("mV", color='white', fontsize=9)

        ax2.set_xlabel("Time (s)", color='white', fontsize=10)
        fig.tight_layout()
        
        # แสดงผลบน Streamlit
        plot_placeholder.pyplot(fig)
        plt.close(fig) # เคลียร์ Memory
        time.sleep(0.01)

# --- 4. ส่วนแสดงผลหน้าหลัก UI ---
st.title("🫀 ECG Clinical Monitoring System")

# ดึงโครงสร้างโฟลเดอร์อัตโนมัติ
structure = {}
try:
    folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f)) and not f.startswith(('.', '__'))]
    for f in folders:
        files = [fl.replace('.py', '') for fl in os.listdir(os.path.join(BASE_DIR, f)) if fl.endswith('.py')]
        if files: structure[f] = sorted(files)
except Exception as e:
    st.error(f"Error reading directory: {e}")

if not structure:
    st.warning("ไม่พบโฟลเดอร์หรือไฟล์โรค (.py) กรุณาตรวจสอบโครงสร้างโฟลเดอร์")
else:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sel_folder = st.radio("📂 หมวดหมู่โรค:", list(structure.keys()))
    with col2:
        sel_disease = st.radio("🧬 เลือกโรคที่ต้องการตรวจ:", structure[sel_folder])
    with col3:
        if sel_disease:
            st.success(f"คัดเลือก: **{sel_disease}**")
            st.write("กดปุ่มด้านล่างเพื่อเปิดหน้าจอมอนิเตอร์วิเคราะห์คลื่นไฟฟ้าหัวใจแบบ Real-time")
            if st.button("▶️ Start Monitoring"):
                show_comparison_monitor(sel_folder, sel_disease)