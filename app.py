import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np

# 網頁基本設定
st.set_page_config(page_title="運動科技專題 - 球類物件偵測系統", page_icon="⚽", layout="wide")
st.title("⚽ 智慧球類運動影像辨識系統 (AI 增強版)")
st.write("本系統已升級至 YOLOv8m 中型精準模型，並啟動運動物件專屬過濾器。")

@st.cache_resource
def load_model():
    # 🧠 大腦升級：換成中型模型 (Medium)，精準度大幅提升，能辨識棒球細節
    return YOLO('yolov8m.pt')

model = load_model()

# 側邊欄參數調整
st.sidebar.header("🔧 系統參數微調")
conf_threshold = st.sidebar.slider("AI 辨識信心度門檻 (Confidence)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# 定義我們專題「只想要看見」的運動相關標籤清單
SPORTS_CLASSES = ["sports ball", "baseball bat", "baseball glove", "tennis racket", "person"]

st.sidebar.markdown("""
### 💡 專題辨識目標
本系統已過濾雜訊，僅專注偵測：
- 🟢 運動球類 (Sports Ball)
- 🟢 棒球棍 (Baseball Bat)
- 🟢 棒球手套 (Baseball Glove)
- 🟢 網球拍 (Tennis Racket)
- 🟢 運動員/裁判 (Person)
""")

uploaded_file = st.file_uploader("請上傳一張球類運動照片 (支援 JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 原始運動影像")
        st.image(image, use_container_width=True)
        
    with st.spinner("AI 深度學習大腦精準分析中..."):
        # 進行推理
        results = model(img_array, conf=conf_threshold)
        
        # 🎯 核心修改：手動篩選，只保留屬於運動類別的 Bounding Box
        boxes = results[0].boxes
        keep_indices = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            # 如果偵測到的東西是我們想要的球類運動項目，才保留
            if cls_name in SPORTS_CLASSES:
                keep_indices.append(i)
        
        # 覆寫結果，只顯示過濾後的運動物件
        results[0].boxes = boxes[keep_indices]
        
        # 繪製結果圖
        annotated_img = results[0].plot()
        detected_items = [model.names[int(box.cls[0])] for box in results[0].boxes]
            
    with col2:
        st.subheader("🎯 AI 運動物件分析結果")
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
        st.info("ℹ️ 未偵測到特定的球類運動物件。")
