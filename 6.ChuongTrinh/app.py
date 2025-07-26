import sys
sys.dont_write_bytecode = True

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

import streamlit as st
import torch
from Models.object_localization import ResNet50BBox
from Models.classification import ResNet50
from PIL import Image
from io import BytesIO
import time
from torchvision import transforms
from datetime import datetime
from PIL import ImageDraw, ImageFont

# Set up logging timestamp
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Running Streamlit app...")

# Set up page configuration
st.set_page_config(layout="wide", page_title="Object Sense | Deep Learning")

def resize_image(image, max_size):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height), Image.LANCZOS)

def process_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        resized = resize_image(image, 2000)
        return image
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# Logic for Classification
@st.cache_data
def process_classification(image, _model, class_names):
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = inference_transform(image)
    input_batch = input_image.unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.eval()
    with torch.no_grad():
        outputs = _model(input_batch.to(device))
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    return image, class_names[predicted_idx], confidence

def process_detection_and_classification(image, detection_model, classification_model, class_names):
    # Chuẩn bị ảnh cho detection
    detection_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    input_image = detection_transform(image)
    input_batch = input_image.unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dự đoán bounding box
    detection_model.eval()
    with torch.no_grad():
        bbox_preds = detection_model(input_batch.to(device))  # [1, 8]
        bbox_preds = bbox_preds.view(2, 4)  # Reshape thành [2, 4] cho 2 box
    
    # Chuyển đổi bounding box về kích thước ảnh gốc
    orig_width, orig_height = image.size
    bboxes = []
    for box in bbox_preds:
        x_center, y_center, w, h = box
        x_min = int((x_center - w/2) * orig_width)
        y_min = int((y_center - h/2) * orig_height)
        x_max = int((x_center + w/2) * orig_width)
        y_max = int((y_center + h/2) * orig_height)
        # Đảm bảo bounding box nằm trong ảnh
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(orig_width, x_max)
        y_max = min(orig_height, y_max)
        bboxes.append([x_min, y_min, x_max, y_max])
    
    # Vẽ bounding box và đánh số lên ảnh
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    
    # Cố gắng load font, nếu không thì dùng font mặc định
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Font chữ và kích thước
    except:
        font = ImageFont.load_default()  # Font mặc định nếu không tìm thấy arial.ttf
    
    for i, box in enumerate(bboxes):
        # Vẽ bounding box
        draw.rectangle(box, outline="red", width=2)
        # Vẽ số thứ tự (Region 1, Region 2) ở góc trên bên trái của bounding box
        x_min, y_min, _, _ = box
        label = f"Region {i+1}"
        draw.text((x_min, y_min - 25), label, fill="red", font=font)
    
    # Cắt ảnh và phân lớp
    cropped_images = []
    predictions = []
    confidences = []
    
    classification_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classification_model.eval()
    for box in bboxes:
        # Cắt ảnh
        cropped = image.crop(box)
        cropped_images.append(cropped)
        
        # Phân lớp
        input_tensor = classification_transform(cropped).unsqueeze(0)
        with torch.no_grad():
            outputs = classification_model(input_tensor.to(device))
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            predictions.append(class_names[predicted_idx])
            confidences.append(confidence)
    
    return image, img_with_bbox, cropped_images, predictions, confidences

def display_classification(upload, model, class_names):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading image...")
        progress_bar.progress(10)
        
        image_bytes = upload.getvalue() if not isinstance(upload, str) else open(upload, "rb").read()
        
        status_text.text("Processing classification...")
        progress_bar.progress(30)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image, prediction, confidence = process_classification(image, model, class_names)
        
        progress_bar.progress(80)
        status_text.text("Displaying results...")
        
        # Tạo 2 cột để hiển thị ảnh và kết quả
        col1, col2 = st.columns([1, 1])  # Chia đều 2 cột
        
        # Hiển thị ảnh gốc với kích thước cố định
        col1.write(f"### Original Image :camera:")
        col1.image(image, width=300)  # Cố định chiều rộng ảnh là 300px, bạn có thể điều chỉnh
        
        # Hiển thị kết quả phân lớp
        col2.write(f"### Classification Results :bar_chart:")
        col2.write(f"**Prediction:** {prediction}")
        col2.write(f"**Confidence:** {confidence:.2%}")
        
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process classification")

def display_detection(upload, detection_model, classification_model, class_names):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading image...")
        progress_bar.progress(10)
        
        image_bytes = upload.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        status_text.text("Processing detection...")
        progress_bar.progress(30)
        
        orig_img, img_with_bbox, cropped_images, predictions, confidences = process_detection_and_classification(
            image, detection_model, classification_model, class_names
        )
        
        progress_bar.progress(80)
        status_text.text("Displaying results...")
        
        # Tạo 3 cột
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Cột 1: Ảnh gốc
        col1.write("### Original Image :camera:")
        col1.image(orig_img, width=300)
        
        # Cột 2: Ảnh với bounding box và số thứ tự
        col2.write("### Image with Bounding Boxes :frame_with_picture:")
        col2.image(img_with_bbox, width=300)
        
        # Cột 3: Kết quả phân lớp cho 2 vùng
        col3.write("### Classification Results :bar_chart:")
        for i, pred in enumerate(predictions):
            col3.write(f"**Region {i+1}:** {pred}")
        
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process detection")

def classification_page():
    st.write("# Image Classification")
    st.write("""
    Welcome to the Image Classification tool powered by advanced artificial intelligence! This feature allows you to upload any image and leverage cutting-edge deep learning technology to classify its content. Whether you're curious about the objects in your photos or need quick insights into visual data, our AI model, built on ResNet50, will analyze your image, remove its background, and provide a prediction along with a confidence score. Simply upload an image and let the magic of AI reveal what’s inside!
    """)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ['Banh cuon', 'Banh mi', 'Bun bo', 'Bun dau', 'Com tam']  # Thay bằng các lớp thực tế
    model = ResNet50(num_classes=len(class_names)).to(device)

    # Tải trọng số mô hình lên đúng thiết bị
    model.load_state_dict(torch.load('./Weights/classification/best_model[Fine-Tuning].pth', map_location=device))

    MAX_FILE_SIZE = 10 * 1024 * 1024
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error(f"File too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
        else:
            display_classification(my_upload, model, class_names)
    else:
        st.info("Please upload an image to get started!")

def detection_page():
    st.write("# Object Detection")
    st.write("""
    Experience our advanced Object Detection tool! This feature uses a custom ResNet50-based model to detect two objects in your image with bounding boxes, then crops these regions and classifies them using our pre-trained classification model. Upload an image to see the original photo, the detected objects with bounding boxes, and the classification results for each detected region—all displayed in an intuitive three-column layout!
    """)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo cả hai mô hình
    class_names = ['Banh cuon', 'Banh mi', 'Bun bo', 'Bun dau', 'Com tam']
    detection_model = ResNet50BBox().to(device)
    classification_model = ResNet50(num_classes=len(class_names)).to(device)
    
    # Tải trọng số
    detection_model.load_state_dict(torch.load('./Weights/object_detection/best_model_object_detection.pth', map_location=device))
    classification_model.load_state_dict(torch.load('./Weights/classification/best_model[Fine-Tuning].pth', map_location=device))
    
    MAX_FILE_SIZE = 10 * 1024 * 1024
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error(f"File too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
        else:
            display_detection(my_upload, detection_model, classification_model, class_names)
    else:
        st.info("Please upload an image to get started!")

# Main interface
st.sidebar.title("Object Sense")
page = st.sidebar.selectbox("Choose a tool", ["Image Classification", "Object Detection"])
st.sidebar.write("\n")
st.sidebar.write("## Upload and Download :gear:")

if page == "Image Classification":
    classification_page()
elif page == "Object Detection":
    detection_page()

with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized to optimize processing
    - Supported formats: PNG, JPG, JPEG
    - Processing time may vary depending on image size and complexity
    """)