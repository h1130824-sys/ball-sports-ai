import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np

# 網頁基本設定
st.set_page_config(page_title="運動科技專題 - 球類物件偵測系統", page_icon="⚽", layout="wide")
st.title("⚽ 智慧球類運動影像辨識系統 (工程診斷版)")
st.write("本系統已整合 YOLOv8m 模型，並提供完整的後台除錯功能。")

@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()

# 側邊欄控制面板
st.sidebar.header("🔧 系統參數與除錯設定")
conf_threshold = st.sidebar.slider("AI 辨識信心度門檻 (Confidence)", min_value=0.01, max_value=1.0, value=0.15, step=0.05)

# 💡 除錯開關：讓使用者可以切換是否開啟過濾器
filter_mode = st.sidebar.checkbox("啟動運動物件專屬過濾器", value=False)

SPORTS_CLASSES = ["sports ball", "baseball bat", "baseball glove", "tennis racket", "person"]

st.sidebar.markdown("""
### 💡 測試小撇步
1. 如果上傳**純白背景**的去背圖，AI 容易漏判。建議上傳**包含真實球場、草地背景**的照片。
2. 若畫面沒框框，可**取消勾選**過濾器，並把信心度調低（例如 0.1），看看 AI 到底把球看成了什麼！
""")

uploaded_file = st.file_uploader("請上傳一張球類運動照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 原始影像")
        st.image(image, use_container_width=True)
        
    with st.spinner("AI 深度學習大腦分析中..."):
        results = model(img_array, conf=conf_threshold)
        boxes = results[0].boxes
        
        # 根據過濾器開關決定是否篩選類別
        keep_indices = []
        for i, box in enumerate(boxes):
            cls_name = model.names[int(box.cls[0])]
            if not filter_mode or (filter_mode and cls_name in SPORTS_CLASSES):
                keep_indices.append(i)
        
        # 套用篩選
        results[0].boxes = boxes[keep_indices]
        annotated_img = results[0].plot()
        detected_items = [model.names[int(box.cls[0])] for box in results[0].boxes]
            
    with col2:
        st.subheader("🎯 AI 物件分析結果")
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, use_container_width=True)
        
    st.success("🎉 影像特徵辨識完成！")
    
    st.write("### 📊 偵測數據清單：")
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
        st.info("ℹ️ 目前沒有偵測到物件。請嘗試將左側「開啟過濾器」取消勾選，或者把「信心度門檻」拉低，查看 AI 的真實盲區！")
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
