# Object-Detection Food

## Giới thiệu

Dự án này xây dựng các mô hình deep learning để nhận diện và phân loại 5 món ăn phổ biến của Việt Nam từ hình ảnh. Sử dụng mạng nơ-ron tích chập (CNN) với ResNet50 cho cả bài toán classification và object detection.

## Cấu trúc thư mục

- `6.ChuongTrinh/`
  - `app.py`: Chương trình chính để chạy mô hình.
  - `classification_resnet50.ipynb`: Notebook huấn luyện và đánh giá mô hình phân loại.
  - `classification_resnet50[Feature_Extraction].ipynb`: Notebook huấn luyện phân loại với phương pháp trích xuất đặc trưng.
  - `classification_resnet50[Fine_Tuning].ipynb`: Notebook huấn luyện phân loại với phương pháp fine-tuning.
  - `object_detection_resnet50.ipynb`: Notebook huấn luyện và đánh giá mô hình phát hiện đối tượng.
  - `requirements.txt`: Danh sách các thư viện cần thiết.
  - `Bin/`: Chứa các file batch để build và khởi động chương trình.
  - `Data/`: Chứa dữ liệu huấn luyện.
    - `dataset_classification/`: Dữ liệu cho bài toán phân loại.
    - `object-detection/`: Dữ liệu cho bài toán phát hiện đối tượng.
  - `Models/`: Chứa các file định nghĩa mô hình.
    - `classification.py`: Định nghĩa mô hình phân loại.
    - `object_localization.py`: Định nghĩa mô hình phát hiện đối tượng.
  - `Weights/`: Chứa các file trọng số mô hình đã huấn luyện.

## Hướng dẫn cài đặt

1. Cài đặt các thư viện cần thiết:

   ```sh
    Bin/build.bat
   ```

2. Chuẩn bị dữ liệu:

   - Đặt dữ liệu vào các thư mục tương ứng trong `6.ChuongTrinh/Data/`.

3. Huấn luyện hoặc sử dụng mô hình:
   - Chạy các notebook trong `6.ChuongTrinh/` để huấn luyện hoặc kiểm thử mô hình.
   - Sử dụng `app.py` để chạy mô hình nhận diện món ăn.

## Sử dụng

- Để chạy nhận diện món ăn, sử dụng:
  ```sh
  Bin/start.bat
  ```

## Tài liệu

- [classification.py](6.ChuongTrinh/Models/classification.py): Định nghĩa mô hình phân loại.
- [object_localization.py](6.ChuongTrinh/Models/object_localization.py): Định nghĩa mô hình phát hiện đối tượng.
- [classification_resnet50.ipynb](6.ChuongTrinh/classification_resnet50.ipynb): Notebook huấn luyện phân loại.
- [object_detection_resnet50.ipynb](6.ChuongTrinh/object_detection_resnet50.ipynb): Notebook huấn luyện phát hiện đối tượng.

## Liên hệ

Mọi thắc mắc vui lòng liên hệ qua email hoặc tham khảo tài liệu kèm theo.
