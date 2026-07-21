# 📖 HƯỚNG DẪN THỰC HÀNH TỰ TAY VIẾT CODE DỰ ÁN AI (FROM SCRATCH)

Chào mừng bạn đến với thư mục tự học thực hành! Thư mục này được thiết kế để bạn tự tay viết lại toàn bộ logic xử lý PyTorch, bộ nạp dữ liệu Dataset, vòng lặp huấn luyện Training Loop, mã suy luận thực tế Inference và giao diện Dashboard Streamlit.

Mỗi file trong thư mục đã được chuẩn bị sẵn các chú thích `# TODO` để dẫn dắt tư duy của bạn. Dưới đây là hướng dẫn chi tiết từng bước để bạn hoàn thiện dự án.

---

## 🛠️ Bước 1: Khởi tạo và cài đặt môi trường

Mở terminal trong thư mục `defect_detection_from_scratch` và chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
py -m pip install -r requirements.txt
```
*(Nếu môi trường của bạn đã cài đặt trước đó ở project cũ, bước này sẽ diễn ra rất nhanh).*

---

## ⚙️ Bước 2: Hoàn thiện tệp cấu hình `config.py`

Mở tệp [config.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/config.py) và điền các tham số cấu hình:
1.  **MODEL_NAME**: Điền `"microsoft/resnet-18"` (hoặc `"google/vit-tiny-patch16-224"` nếu máy bạn mạnh).
2.  **BATCH_SIZE**: Điền số nguyên `4`.
3.  **EPOCHS**: Điền số nguyên `15`.
4.  **LEARNING_RATE**: Điền tốc độ học `1e-4`.
5.  **N_AUGMENT**: Điền `15` (mỗi ảnh gốc sinh thêm 15 ảnh biến thể).

---

## 🖼️ Bước 3: Hoàn thiện bộ nạp dữ liệu `training/dataset.py`

Mở tệp [training/dataset.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/training/dataset.py):

### 1. Viết các bộ tăng cường dữ liệu OpenCV (`augment_image`)
*   **Lật ngang**: `return cv2.flip(img, 1)`
*   **Lật dọc**: `return cv2.flip(img, 0)`
*   **Xoay góc nhỏ**: Tính ma trận quay $M$:
    ```python
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    ```
*   **Thay đổi độ sáng**: `return cv2.convertScaleAbs(img, alpha=1.15, beta=15)` (sáng) hoặc `alpha=0.85, beta=-15` (tối).

### 2. Viết logic lấy mẫu `__getitem__`
*   Tính chỉ số ảnh gốc và chỉ số biến thể:
    ```python
    if self.augment:
        base_idx = idx // (self.n_augment + 1)
        aug_idx = idx % (self.n_augment + 1)
    else:
        base_idx = idx
        aug_idx = 0
    ```
*   Biến đổi ảnh OpenCV sang ảnh PIL RGB để đưa vào Hugging Face:
    ```python
    img_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    ```
*   Chuẩn hóa tensor thông qua `hf_image_processor`:
    ```python
    inputs = self.hf_image_processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].squeeze(0)
    ```

---

## 📈 Bước 4: Hoàn thiện tệp huấn luyện `training/train_hf.py`

Mở tệp [training/train_hf.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/training/train_hf.py):

### 1. Tải Model & Processor
*   Tải bộ xử lý: `hf_image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)`
*   Tải mô hình phân loại:
    ```python
    model = AutoModelForImageClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.CLASS_NAMES),
        id2label={i: c for i, c in enumerate(config.CLASS_NAMES)},
        label2id={c: i for i, c in enumerate(config.CLASS_NAMES)},
        ignore_mismatched_sizes=True
    )
    ```

### 2. Thiết lập đóng băng trọng số (Freeze)
*   Đóng băng toàn bộ mô hình:
    ```python
    for param in model.parameters():
        param.requires_grad = False
    ```
*   Mở khóa lớp classifier cuối cùng (ví dụ trên `resnet-18` là lớp `classifier`):
    ```python
    for param in model.classifier.parameters():
        param.requires_grad = True
    ```

### 3. Thiết lập Optimizer & Loss
*   Tạo tối ưu hóa AdamW:
    ```python
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    ```

### 4. Viết vòng lặp huấn luyện PyTorch
Trong vòng lặp loader, viết 6 dòng code vàng của PyTorch:
```python
optimizer.zero_grad()                                # 1. Reset gradient về 0
outputs = model(pixel_values=pixel_values)           # 2. Forward pass
loss = criterion(outputs.logits, batch_labels)       # 3. Tính Loss
loss.backward()                                      # 4. Backward pass (tính đạo hàm)
optimizer.step()                                     # 5. Cập nhật trọng số
```

### 5. Lưu mô hình đã train
```python
model.save_pretrained(config.MODEL_DIR)
hf_image_processor.save_pretrained(config.MODEL_DIR)
```

---

## 🔍 Bước 5: Hoàn thiện tệp suy luận `inference/predict.py`

Mở tệp [inference/predict.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/inference/predict.py):
*   **Tiền xử lý**: Gọi `proc_result = self.opencv_processor.process(img_bgr)` để lấy ROI.
*   **Đổi định dạng**: Chuyển `proc_result.roi` (BGR) sang RGB và bọc bằng `Image.fromarray()`.
*   **Chuẩn hóa**: Chạy qua `self.image_processor(images=pil_img, return_tensors="pt")` để nhận dạng tensor.
*   **Dự đoán**:
    ```python
    with torch.no_grad():
        outputs = self.model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)
    ```
*   Trích xuất chỉ số lớp và % độ tin cậy:
    ```python
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    ```

---

## 🌐 Bước 6: Hoàn thiện tệp Dashboard `app.py`

Mở tệp [app.py](file:///d:/Cole/AI%20engineer%20LV%20up/defect_detection_from_scratch/app.py):
*   Gọi hàm suy luận: `result = inspector.inspect(img_bgr)`
*   Hiển thị Banner: `st.success()` nếu kết quả là OK, `st.error()` nếu là NG.
*   Hiển thị ảnh và đồ thị: Dùng `st.columns` để chia layout, `st.image()` để vẽ ảnh và `st.progress()` để vẽ xác suất của mô hình.
*   Hiển thị 6 ảnh OpenCV: Duyệt qua `result["vis_steps"]` để in ra 6 bước xử lý ảnh.

---

## 💡 Mẹo khi gặp lỗi hoặc bí ý tưởng:
Nếu bạn gặp bất kỳ khó khăn nào trong quá trình thực hành, hãy mở thư mục dự án hoàn chỉnh **`defect_detection_huggingface`** để đối chiếu và xem code mẫu nhé! Mục đích là giúp bạn tự gõ lại để nhớ cú pháp và hiểu tư duy kỹ thuật của PyTorch & Hugging Face.
