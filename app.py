import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np

# 網頁基本設定
st.set_page_config(page_title="運動科技專題 - 球類物件偵測系統", page_icon="⚽", layout="wide")
st.title("⚽ 智慧球類運動影像辨識系統")
st.write("本系統採用 YOLOv8 深度學習模型，專為運動賽事、球類與運動裝備進行即時偵測與數據統計。")

@st.cache_resource
def load_model():
    # 強制載入官方最穩定的 8 格式權重
    return YOLO('yolov8n.pt')

model = load_model()

# 側邊欄參數調整
st.sidebar.header("🔧 系統參數微調")
conf_threshold = st.sidebar.slider("AI 辨識信心度門檻 (Confidence)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("""
### 💡 專題測試指引
本系統預設支援以下球類與裝備：
- 🟢 **sports ball** (各式球類)
- 🟢 **baseball bat** (棒球棍)
- 🟢 **baseball glove** (棒球手套)
- 🟢 **tennis racket** (網球拍)
""")

uploaded_file = st.file_uploader("請上傳一張球類運動照片 (支援 JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. 讀取圖片並強制確保是標準 RGB 3通道（防呆關鍵！）
    image = PIL.Image.open(uploaded_file).convert("RGB")
    
    # 2. 轉為 numpy 矩陣
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 原始運動影像")
        st.image(image, use_container_width=True)
        
    with st.spinner("AI 深度學習大腦分析中..."):
        # 3. 確保送入模型的是乾淨的 RGB 矩陣
        results = model(img_array, conf=conf_threshold)
        annotated_img = results[0].plot()
        boxes = results[0].boxes
        detected_items = [model.names[int(box.cls[0])] for box in boxes]
            
    with col2:
        st.subheader("🎯 AI 運動物件分析結果")
        # 4. YOLO plot 吐出來的是 BGR，必須轉回 RGB 網頁才不會綠綠的
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, use_container_width=True)
        
    st.success("🎉 影像特徵辨識完成！")
    
    st.write("### 📊 賽事裝備與球類數量統計：")
    if detected_items:
        item_counts = {x: detected_items.count(x) for x in set(detected_items)}
        for item, count in item_counts.items():
            zh_name = item
            if item == "sports ball": zh_name = "⚽ 運動球類 (Sports Ball)"
            elif item == "baseball bat": zh_name = "🏏 棒球棍 (Baseball Bat)"
            elif item == "baseball glove": zh_name = "⚾ 棒球手套 (Baseball Glove)"
            elif item == "tennis racket": zh_name = "🎾 網球拍 (Tennis Racket)"
            elif item == "person": zh_name = "🏃‍♂️ 運動員/裁判 (Person)"
            st.write(f"- **{zh_name}**: {count} 個")
    else:
        st.info("ℹ️ 在當前信心度門檻下，未偵測到特定的球類運動物件。請嘗試調低左側門檻再試一次！")
