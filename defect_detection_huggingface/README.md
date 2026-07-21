# 🔍 Hệ Thống Nhận Diện Sản Phẩm Lỗi — Computer Vision (Hugging Face + PyTorch)

Dự án này là một hệ thống kiểm tra chất lượng sản phẩm tự động (Quality Control) được xây dựng theo định hướng học tập và thực hành của một **AI Engineer**. Hệ thống sử dụng kết hợp giữa **Computer Vision truyền thống (OpenCV)** để tiền xử lý/cắt vùng đặc trưng (ROI) và **Deep Learning (Hugging Face + PyTorch)** để phân loại lỗi sản phẩm (OK/NG).

---

## 🎯 Định Hướng Lập Trình: AI Engineer vs. Machine Learning Researcher

*   **Lập trình thực tế**: Thay vì thiết kế một kiến trúc mạng nơ-ron tích chập (Custom CNN) từ đầu và huấn luyện trên dữ liệu cục bộ từ số không (đòi hỏi hàng nghìn ảnh và sức mạnh tính toán khổng lồ), một **AI Engineer** tập trung vào việc **tận dụng tài nguyên có sẵn**:
    *   Tải mô hình đã huấn luyện trước (**Pre-trained Models**) trên hàng triệu ảnh từ thư viện **Hugging Face Hub**.
    *   Sử dụng phương pháp **Transfer Learning (Học chuyển giao)**: đóng băng phần thân của mô hình và chỉ huấn luyện lại phân lớp cuối cùng (**classification head**) cho bài toán cụ thể.
    *   Kế thừa và tích hợp các pipeline tiền xử lý ảnh OpenCV đã được tối ưu hóa.
    *   Xây dựng hệ thống hoàn chỉnh từ Tiền xử lý -> Huấn luyện -> Suy luận -> Triển khai Dashboard Web.

---

## 🤖 Báo Cáo Tự Học: Lựa Chọn Mô Hình & Các Giải Pháp Thay Thế

Dự án này mặc định sử dụng mô hình **ResNet-18** (`microsoft/resnet-18`) của Microsoft thông qua Hugging Face. Dưới đây là lý do lựa chọn và so sánh chi tiết với các kiến trúc khác:

### 1. Lý do chọn ResNet-18 làm mô hình mặc định:
*   **Hiệu quả truyền tải đặc trưng**: Mạng ResNet sử dụng các khối kết nối tắt (Skip Connections) giúp giải quyết triệt để lỗi triệt tiêu gradient (vanishing gradient) khi mô hình sâu hơn. ResNet-18 dù nông nhưng vẫn trích xuất cực tốt các đặc trưng về hình học, kết cấu bề mặt, và màu sắc của cos đồng.
*   **Tối ưu tài nguyên CPU**: Trong môi trường tự học trên máy cá nhân không có GPU rời, việc huấn luyện và chạy suy luận của ResNet-18 cực kỳ nhanh và mượt mà, không xảy ra hiện tượng tràn bộ nhớ RAM/VRAM.
*   **Kháng Overfitting trên tập dữ liệu nhỏ**: Do tập dữ liệu mẫu chỉ gồm 3 ảnh OK và 3 ảnh NG, các mô hình lớn rất dễ bị "thuộc lòng" dữ liệu (overfit). ResNet-18 có kích thước tham số nhỏ (~11 triệu tham số), kết hợp với kỹ thuật đóng băng trọng số, sẽ giúp mô hình có khả năng tổng quát hóa tốt hơn nhiều.

### 2. Các mô hình thay thế có thể sử dụng:

Bạn có thể thay đổi mô hình dễ dàng bằng cách đổi tên trong tệp `config.py`:

| Mô hình | Model ID trên Hugging Face | Số lượng tham số | Ưu điểm chính | Nhược điểm / Hạn chế |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** *(Đang chọn)* | `microsoft/resnet-18` | ~11M | Cực kỳ nhẹ, chạy mượt trên CPU, huấn luyện siêu nhanh, khó bị overfit khi ít dữ liệu. | Độ chính xác tối đa thấp hơn các dòng Transformer trên các bài toán phức tạp cao. |
| **Vision Transformer (ViT)** | `google/vit-tiny-patch16-224` | ~5.7M | Hiểu mối quan hệ không gian toàn cục của ảnh bằng cơ chế Self-Attention. | Huấn luyện chậm trên CPU, rất dễ bị overfit nếu dữ liệu huấn luyện quá ít. |
| **ConvNeXt** | `facebook/convnext-tiny-224` | ~28M | Kiến trúc CNN hiện đại, tích hợp tư duy thiết kế của Transformer. Đạt độ chính xác rất cao. | Nặng hơn ResNet-18, đòi hỏi CPU mạnh hơn một chút. |
| **MobileNetV2** | `google/mobilenet_v2_1.0_224` | ~3.5M | Siêu nhẹ, thiết kế chuyên biệt cho thiết bị di động và các hệ thống nhúng Edge Devices. | Độ chính xác giảm nhẹ đối với các lỗi sản phẩm tinh vi, khó nhìn. |

---

## 🛠️ Quy Trình Pipeline Xử Lý Ảnh (Kết hợp OpenCV + Deep Learning)

Hệ thống kết hợp sức mạnh của thị giác máy tính truyền thống và học sâu qua các bước:

```
[Ảnh gốc đầu vào] 
        │
        ▼ (OpenCV 6 bước tiền xử lý)
1. Tách nền -> 2. Định biên -> 3. Xoay thẳng -> 4. Cắt chuẩn hóa -> 5. Vùng đồng -> 6. Cắt ROI 
        │
        ▼ (Ảnh ROI kích thước 224x224)
[Hugging Face ImageProcessor] (Chuẩn hóa mean, std, đổi sang RGB Tensor)
        │
        ▼ 
[Mô hình PyTorch Fine-tuned] (Forward Pass tính điểm số Logits)
        │
        ▼
[Bộ phân loại & Áp ngưỡng Threshold] (Đưa ra kết quả cuối cùng OK/NG kèm % độ tin cậy)
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
defect_detection_huggingface/
├── README.md                 # Hướng dẫn học tập, so sánh mô hình và cách chạy
├── requirements.txt         # Khai báo các thư viện phụ thuộc (numpy<2, torch, transformers, ...)
├── config.py                 # Cấu hình tập trung (Model ID, Learning Rate, Epoch, Paths)
├── run_demo.bat             # File thực thi tương tác CLI (Cài đặt, Huấn luyện, Suy luận, Web App)
├── run_app.bat              # File chạy trực tiếp Dashboard Streamlit bằng 1-click
│
├── preprocessing/            # Pipeline OpenCV trích xuất ROI sản phẩm
│   ├── __init__.py
│   └── image_processor.py    # Class xử lý ảnh 6 bước
│
├── training/                 # Mã nguồn huấn luyện
│   ├── __init__.py
│   ├── dataset.py            # Dataset PyTorch + Data Augmentation động + HF ImageProcessor
│   └── train_hf.py           # Tinh chỉnh (Fine-tuning) mô hình pre-trained bằng PyTorch
│
├── inference/                # Mã nguồn chạy thực tế
│   ├── __init__.py
│   └── predict.py            # Dự đoán ảnh gốc: OpenCV + Hugging Face Model
│
└── app.py                    # Giao diện Web Dashboard trực quan bằng Streamlit
```

---

## 🚀 Cài Đặt & Sử Dụng

### Cách 1: Sử dụng File Chạy Tiện Ích `run_demo.bat` (Khuyến nghị)
Bạn chỉ cần click đúp vào file `run_demo.bat` trên Windows, một menu tương tác bằng tiếng Việt sẽ hiện ra giúp bạn thực hiện mọi thao tác:
1. Nhấn `4` để tự động cài đặt các thư viện cần thiết.
2. Nhấn `1` để bắt đầu huấn luyện tinh chỉnh mô hình Hugging Face.
3. Nhấn `2` để chạy suy luận thử nghiệm trên các ảnh gốc.
4. Nhấn `3` để mở Dashboard Web Streamlit.

---

### Cách 2: Chạy trực tiếp bằng dòng lệnh (Dùng command `py` cho Windows)

Môi trường hiện tại sử dụng công cụ quản lý Python `py`. Dưới đây là các câu lệnh chạy thủ công:

#### 1. Cài đặt các thư viện phụ thuộc:
```bash
py -m pip install -r requirements.txt
```
*(requirements.txt đã được khóa phiên bản `numpy<2` để tránh xung đột với OpenCV trên Windows).*

#### 2. Huấn luyện mô hình:
```bash
py training/train_hf.py
```
*Script sẽ tự động đọc các ảnh mẫu trong `training_images/labeled/OK` và `training_images/labeled/NG`, thực hiện tăng cường dữ liệu (Augmentation), huấn luyện lớp phân loại cuối cùng, vẽ đồ thị huấn luyện lưu tại `output/training_curves.png` và lưu model đã fine-tune tại `output/model/`.*

#### 3. Chạy suy luận trên ảnh gốc:
```bash
# Chạy suy luận trên toàn bộ ảnh gốc mặc định trong raw_images/
py inference/predict.py

# Chạy suy luận trên 1 ảnh chỉ định
py inference/predict.py --image "d:\Cole\AI engineer LV up\raw_images\z7958188267677_5f8ea55ba285f6c7b998bc54ad56b9f5.jpg"
```
*Kết quả suy luận trực quan (gồm bounding box bao quanh sản phẩm, nhãn phân loại OK/NG và độ tin cậy) sẽ được lưu tại thư mục `output/predictions/`.*

#### 4. Khởi động Web Dashboard:
```bash
streamlit run app.py
```
*Giao diện dashboard sẽ tự động được mở trong trình duyệt của bạn tại địa chỉ: `http://localhost:8501`. Bạn có thể tải ảnh lên hoặc chọn ảnh mẫu, xem trực quan kết quả tiền xử lý ảnh 6 bước bằng OpenCV cùng kết quả phân loại của mô hình học sâu.*
