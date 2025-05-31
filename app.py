import streamlit as st
import requests
import torch
from PIL import Image
from io import BytesIO
from VQA import VQAModel

API_URL = "https://07f7-34-142-241-247.ngrok-free.app/vqa"
CHECKPOINT_PATH = "./final_model.pt"

st.set_page_config(
    page_title="VQA - Visual Question Answering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4527A0;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        border-bottom: 2px solid #9575CD;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 1rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #F3F4F6;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #4527A0;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #9575CD;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #6200EA;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #4527A0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    div.block-container {
        padding-top: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-left: 5px solid #2196F3;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-left: 5px solid #F44336;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 🔄 Chọn model")
    processing_mode = st.radio("", ("VQA training", "VQA fine-tuning"), 
                               help="Chọn xử lý với mô hình được training hoặc qua model được fine-tuning")
    
    device = "cpu"  
    if processing_mode == "VQA training":
        st.markdown("### 💻 Thiết bị xử lý")
        device_chosen = st.radio("", ("CPU", "CUDA"), 
                          help="Chọn CPU hoặc CUDA (nếu có GPU hỗ trợ)")
        
        is_cuda_available = torch.cuda.is_available()
        device = "cuda" if device_chosen == "CUDA" and is_cuda_available else "cpu"
        
        if device == "cuda":
            st.markdown(f'<div class="success-box">✅ Đang sử dụng GPU: {torch.cuda.get_device_name(0)}</div>', unsafe_allow_html=True)
        else:
            if device_chosen == "CUDA" and not is_cuda_available:
                st.markdown('<div class="error-box">⚠️ CUDA không khả dụng, sẽ sử dụng CPU thay thế</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">ℹ️ Đang sử dụng CPU</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🖼️ Phương thức nhập hình ảnh")
    input_method = st.radio("", ("Tải lên", "URL"), 
                            help="Chọn cách tải lên hình ảnh")
    
    st.markdown("---")
    
    st.markdown("### 📌 Về dự án")
    st.markdown("""
    <div class="info-box">
    Dự án hỏi đáp với hình ảnh sử dụng hai model
    <p>1. Model được xây dựng lại từ đầu với Pho BERT - ViT - BART Pho.</p> 
    <p>2. Mô hình Qwen2-VL-2B được fine-tuned sử dụng Unsloth.</p>
    </div>
    """, unsafe_allow_html=True)

# MAIN UI
st.markdown('<h1 class="main-header">🔍 Visual Question Answering (VQA)</h1>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Hỏi đáp thông minh với hình ảnh - sử dụng mô hình Qwen2-VL-2B</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

def call_vqa_local(image, question, device): 
    try:
        vqamodel = VQAModel()
        vqamodel.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device(device)))
        vqamodel.eval()
        answer = vqamodel.generate_answer(image, question)
        return answer
    except Exception as e:
        return f"Lỗi khi Mark: Lỗi khi xử lý local: {str(e)}"

def call_vqa_api(image, question):
    try:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        data = {'question': question}
        response = requests.post(API_URL, files=files, data=data)

        if response.status_code == 200:
            return response.json().get('answer', 'Không nhận được câu trả lời')
        else:
            return f"Lỗi: {response.json().get('error', 'Không xác định')}"
    except Exception as e:
        return f"Lỗi khi gọi API: {str(e)}"

with col1:
    st.markdown('<h3 style="color: #5E35B1;">📷 Hình ảnh</h3>', unsafe_allow_html=True)
    
    image = None
    if input_method == "Tải lên":
        uploaded_file = st.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)
    else:
        url = st.text_input("Nhập URL hình ảnh:", "https://cdnv2.tgdd.vn/mwg-static/common/News/1570960/4-1280x720.jpg")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Hình ảnh từ URL", use_column_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Lỗi khi tải hình ảnh từ URL: {e}</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h3 style="color: #5E35B1;">❓ Đặt câu hỏi</h3>', unsafe_allow_html=True)
    question = st.text_input("Nhập câu hỏi về hình ảnh:", "Hình ảnh mô tả điều gì?")
    
    submit_button = st.button("🚀 Gửi câu hỏi")

    if submit_button:
        if image is None:
            st.markdown('<div class="error-box">⚠️ Vui lòng cung cấp một hình ảnh!</div>', unsafe_allow_html=True)
        elif not question:
            st.markdown('<div class="error-box">⚠️ Vui lòng nhập câu hỏi!</div>', unsafe_allow_html=True)
        else:
            with st.spinner("🕒 Đang xử lý câu hỏi..."):
                if processing_mode == "VQA training":
                    answer = call_vqa_local(image, question, device)
                else:
                    answer = call_vqa_api(image, question)
                answer = answer.split("assistant")[-1]
                
                st.markdown(f'''
                    <div class="result-container">
                        <p><strong>📝 Câu hỏi:</strong> {question}</p>
                        <p><strong>🔍 Câu trả lời:</strong></p>
                        <p style="background-color: white; padding: 15px; border-radius: 5px;">{answer}</p>
                    </div>
                ''', unsafe_allow_html=True)