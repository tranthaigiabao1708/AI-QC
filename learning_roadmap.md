# 🗺️ Lộ Trình 10 Bài Toán Nhỏ Tự Tay Viết Code AI QC (From Scratch)

Để giúp bạn chinh phục dự án kiểm tra chất lượng (QC) này một cách tự tin, chúng ta sẽ chia dự án lớn thành **10 bài toán nhỏ**. Mỗi bài toán tập trung giải quyết một kỹ năng duy nhất, đi từ cơ bản đến nâng cao. 

Khi hoàn thành bài toán 10, bạn sẽ có một hệ thống AI hoàn chỉnh chạy mượt mà.

---

## 🗺️ Tóm tắt lộ trình 10 bài toán nhỏ

```
[BÀI TOÁN 1] Cấu hình dự án (config.py)
      │
      ▼
[BÀI TOÁN 2] Viết hàm Augmentation đơn lẻ bằng OpenCV
      │
      ▼
[BÀI TOÁN 3] Xây dựng Dataset PyTorch đọc ảnh cơ bản
      │
      ▼
[BÀI TOÁN 4] Tích hợp bộ tiền xử lý Hugging Face vào Dataset
      │
      ▼
[BÀI TOÁN 5] Tải mô hình Hugging Face & Khóa trọng số (Freeze)
      │
      ▼
[BÀI TOÁN 6] Viết bước học thử nghiệm (1 batch Forward/Backward)
      │
      ▼
[BÀI TOÁN 7] Hoàn thiện vòng lặp huấn luyện (Full Training Loop)
      │
      ▼
[BÀI TOÁN 8] Tính toán số liệu đánh giá mô hình chuyên nghiệp
      │
      ▼
[BÀI TOÁN 9] Viết bộ suy luận thực tế (OpenCV ROI + AI Inference)
      │
      ▼
[BÀI TOÁN 10] Triển khai Web Dashboard tương tác bằng Streamlit
```

---

## 📖 Chi tiết từng bài toán & Nhiệm vụ thực hành

### 📍 Bài toán 1: Thiết lập cấu hình hệ thống (`config.py`)
*   **Mục tiêu**: Làm quen với cấu trúc thư mục dự án và khai báo các hằng số.
*   **Nhiệm vụ**: Điền các siêu tham số học tập (`MODEL_NAME`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `N_AUGMENT`) vào file [config.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/config.py).
*   **Ý nghĩa**: Giúp code của bạn sạch sẽ, dễ thay đổi cấu hình mà không cần sửa code logic.

### 📍 Bài toán 2: Viết các phép biến đổi ảnh bằng OpenCV (`augment_image`)
*   **Mục tiêu**: Hiểu cách tăng cường dữ liệu ảnh bằng code OpenCV cơ bản.
*   **Nhiệm vụ**: Vào [training/dataset.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/training/dataset.py), viết các hàm:
    *   Lật ảnh (ngang, dọc) bằng `cv2.flip`.
    *   Xoay ảnh một góc nhẹ bằng `cv2.getRotationMatrix2D` và `cv2.warpAffine`.
    *   Tăng giảm độ sáng bằng `cv2.convertScaleAbs`.
*   **Ý nghĩa**: Giúp nhân bản số lượng dữ liệu huấn luyện khi tập ảnh gốc quá nhỏ.

### 📍 Bài toán 3: Xây dựng lớp Dataset PyTorch cơ bản
*   **Mục tiêu**: Hiểu cách PyTorch quản lý luồng đọc dữ liệu từ đĩa cứng.
*   **Nhiệm vụ**: Viết lớp `DefectDataset(Dataset)` nhận danh sách file ảnh và nhãn (0 cho OK, 1 cho NG). Viết hàm `__len__` trả về tổng số ảnh và `__getitem__` đọc ảnh gốc tương ứng bằng OpenCV.
*   **Ý nghĩa**: Nền tảng của mọi dự án PyTorch, giúp nạp dữ liệu một cách tuần tự.

### 📍 Bài toán 4: Tích hợp bộ chuẩn hóa của Hugging Face vào Dataset
*   **Mục tiêu**: Chuyển đổi định dạng ảnh OpenCV sang chuẩn Tensor của mô hình Hugging Face.
*   **Nhiệm vụ**: Trong `__getitem__`, chuyển đổi ảnh OpenCV (BGR) sang PIL Image (RGB). Sau đó, sử dụng đối tượng `hf_image_processor` để chuyển đổi ảnh PIL thành Tensor chuẩn hóa (scale pixel về 0-1, áp dụng mean và std của ImageNet).
*   **Ý nghĩa**: Đồng bộ hóa định dạng ảnh đầu vào chính xác như lúc mô hình pre-trained được huấn luyện.

### 📍 Bài toán 5: Tải mô hình Hugging Face và Khóa trọng số (Transfer Learning Setup)
*   **Mục tiêu**: Thực hành kỹ năng Học chuyển giao của AI Engineer.
*   **Nhiệm vụ**: Mở file [training/train_hf.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/training/train_hf.py). Viết code sử dụng `AutoModelForImageClassification.from_pretrained` để tải ResNet-18. Viết vòng lặp đóng băng toàn bộ tham số của mô hình (`requires_grad = False`) và chỉ mở khóa lớp `model.classifier` (`requires_grad = True`).
*   **Ý nghĩa**: Đảm bảo mô hình chạy cực nhẹ, train cực nhanh trên CPU và không bị quá khớp (overfit).

### 📍 Bài toán 6: Viết bước huấn luyện thử nghiệm (1 Batch)
*   **Mục tiêu**: Hiểu cơ chế hoạt động của thuật toán Lan truyền ngược (Backpropagation) trong PyTorch.
*   **Nhiệm vụ**: Viết một đoạn code nhỏ để chạy thử 1 batch dữ liệu qua mô hình:
    *   Dùng `optimizer.zero_grad()` xóa đạo hàm cũ.
    *   Chạy `outputs = model(pixel_values=...)` để lấy dự đoán.
    *   Tính Loss bằng `criterion(logits, labels)`.
    *   Chạy `loss.backward()` để tính đạo hàm.
    *   Chạy `optimizer.step()` để cập nhật trọng số.
*   **Ý nghĩa**: Nắm vững trục cốt lõi của việc dạy mô hình học.

### 📍 Bài toán 7: Hoàn thiện vòng lặp huấn luyện (Full Training Loop)
*   **Mục tiêu**: Kiểm soát quá trình huấn luyện qua nhiều Epochs.
*   **Nhiệm vụ**: Kết hợp bước 6 vào vòng lặp nhiều epoch. Theo dõi chỉ số Loss và Accuracy trung bình của mỗi epoch. Vẽ đồ thị huấn luyện bằng `matplotlib` và lưu thành file `training_curves.png`.
*   **Ý nghĩa**: Giúp bạn nhìn trực quan xem mô hình có thực sự "học" được gì không (Loss giảm dần, Accuracy tăng dần).

### 📍 Bài toán 8: Đánh giá mô hình chuyên nghiệp (Evaluation Metrics)
*   **Mục tiêu**: Sử dụng các chỉ số đo lường chuẩn công nghiệp để đánh giá mô hình.
*   **Nhiệm vụ**: Chuyển mô hình sang chế độ đánh giá (`model.eval()`). Sử dụng `scikit-learn` để tính toán ma trận nhầm lẫn (`confusion_matrix`) và in bảng báo cáo chi tiết (`classification_report` gồm Precision, Recall, F1-Score).
*   **Ý nghĩa**: Giúp bạn biết mô hình có bị thiên vị (bias) hoặc hay đoán nhầm lớp nào nhất.

### 📍 Bài toán 9: Xây dựng pipeline suy luận thực tế (Inference Pipeline)
*   **Mục tiêu**: Tạo ra một sản phẩm suy luận hoàn chỉnh từ ảnh gốc.
*   **Nhiệm vụ**: Mở file [inference/predict.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/inference/predict.py), viết hàm kết hợp: Nhận ảnh gốc $\rightarrow$ Chạy qua bộ tiền xử lý OpenCV để cắt ROI $\rightarrow$ Chuẩn hóa ROI nạp vào mô hình AI $\rightarrow$ Lấy kết quả $\rightarrow$ Vẽ hộp bounding box và nhãn (OK/NG) lên ảnh gốc.
*   **Ý nghĩa**: Đây chính là lõi chạy thực tế khi bàn giao dự án cho khách hàng hoặc nhà máy.

### 📍 Bài toán 10: Dựng Web Dashboard tương tác bằng Streamlit
*   **Mục tiêu**: Tạo giao diện người dùng trực quan, chuyên nghiệp.
*   **Nhiệm vụ**: Mở file [app.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/app.py). Dựng giao diện Web cho phép upload ảnh, hiển thị kết quả phân loại lớn, hiển thị biểu đồ xác suất và vẽ trực quan 6 bước trung gian của OpenCV.
*   **Ý nghĩa**: Biến toàn bộ code dòng lệnh khô khan thành một sản phẩm Web lung linh, dễ dàng trình diễn (demo) cho nhà tuyển dụng hoặc khách hàng.
