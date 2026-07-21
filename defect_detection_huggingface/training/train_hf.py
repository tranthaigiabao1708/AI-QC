import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from loguru import logger
import numpy as np

# Thêm đường dẫn gốc vào python path để import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from dataset import DefectDataset

# Thiết lập logger ghi ra file và console
logger.add(config.OUTPUT_DIR / "training.log", rotation="500 MB", encoding="utf-8")

def get_image_files(directory):
    """Lấy danh sách tất cả các đường dẫn ảnh trong thư mục."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    directory = Path(directory)
    if not directory.exists():
        return []
    return [f for f in directory.iterdir() if f.suffix.lower() in exts]

def main():
    logger.info("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MODEL HUGGING FACE ===")
    
    # 1. Thu thập dữ liệu
    ok_dir = config.DATA_DIR / "OK"
    ng_dir = config.DATA_DIR / "NG"
    
    ok_files = get_image_files(ok_dir)
    ng_files = get_image_files(ng_dir)
    
    logger.info(f"Tìm thấy {len(ok_files)} ảnh OK và {len(ng_files)} ảnh NG")
    
    if len(ok_files) == 0 or len(ng_files) == 0:
        logger.error(f"Thiếu dữ liệu! Hãy đảm bảo thư mục {ok_dir} và {ng_dir} có ảnh.")
        logger.error("Bạn có thể copy ảnh từ thư mục 'training_images/all_rois' sang để test.")
        sys.exit(1)
        
    # Tạo danh sách file và nhãn (0: OK, 1: NG)
    file_paths = ok_files + ng_files
    labels = [0] * len(ok_files) + [1] * len(ng_files)
    
    # 2. Tải bộ tiền xử lý ảnh từ Hugging Face
    logger.info(f"Đang tải Hugging Face Image Processor cho mô hình: {config.MODEL_NAME}...")
    try:
        hf_image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
    except Exception as e:
        logger.error(f"Lỗi tải Image Processor: {e}")
        sys.exit(1)
        
    # 3. Tạo Dataset và DataLoader
    # Sử dụng Data Augmentation để tăng lượng ảnh huấn luyện
    train_dataset = DefectDataset(
        image_paths=file_paths,
        labels=labels,
        hf_image_processor=hf_image_processor,
        augment=True,
        n_augment=config.N_AUGMENT
    )
    
    # Tạo dataset đánh giá (không augment) để kiểm tra độ chính xác trên dữ liệu gốc
    eval_dataset = DefectDataset(
        image_paths=file_paths,
        labels=labels,
        hf_image_processor=hf_image_processor,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 4. Tải mô hình từ Hugging Face
    logger.info(f"Đang tải mô hình pre-trained: {config.MODEL_NAME}...")
    try:
        model = AutoModelForImageClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=len(config.CLASS_NAMES),
            id2label={i: c for i, c in enumerate(config.CLASS_NAMES)},
            label2id={c: i for i, c in enumerate(config.CLASS_NAMES)},
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        logger.error(f"Lỗi tải mô hình: {e}")
        sys.exit(1)
        
    # 5. Đóng băng backbone và chỉ huấn luyện lớp Classifier cuối cùng (Transfer Learning)
    # Đây là thực hành chuẩn của AI Engineer khi tập dữ liệu nhỏ
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở khóa gradient cho classifier head
    classifier_found = False
    for name, module in model.named_children():
        if 'classifier' in name or 'head' in name:
            logger.info(f"Mở khóa huấn luyện cho phân lớp cuối cùng (classifier): {name}")
            for param in module.parameters():
                param.requires_grad = True
            classifier_found = True
            
    if not classifier_found:
        # Dự phòng: mở khóa module con cuối cùng của mô hình
        last_name = list(model.named_children())[-1][0]
        logger.info(f"Mở khóa huấn luyện cho module con cuối cùng: {last_name}")
        for param in list(model.children())[-1].parameters():
            param.requires_grad = True
            
    # Di chuyển mô hình lên thiết bị tính toán (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Huấn luyện trên thiết bị: {device}")
    
    # 6. Thiết lập bộ tối ưu hóa (Optimizer) và Hàm tính Loss (Loss Function)
    # Chỉ tối ưu hóa những tham số có yêu cầu tính gradient (requires_grad = True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 7. Vòng lặp huấn luyện (Training Loop)
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            loss = criterion(logits, batch_labels)
            
            # Backward pass & cập nhật trọng số
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * pixel_values.size(0)
            
            # Tính toán độ chính xác trong batch
            _, preds = torch.max(logits, 1)
            correct_preds += torch.sum(preds == batch_labels).item()
            total_preds += batch_labels.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        
        logger.info(f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1%}")
        
    logger.info("Hoàn tất quá trình huấn luyện!")
    
    # 8. Đánh giá mô hình trên dữ liệu gốc (Eval mà không dùng Augmentation)
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            _, preds = torch.max(outputs.logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
            
    # Tính các chỉ số đánh giá bằng scikit-learn
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=config.CLASS_NAMES, zero_division=0)
    
    logger.info("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN DỮ LIỆU GỐC ===")
    logger.info(f"Accuracy: {acc:.1%}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    
    # 9. Lưu mô hình và bộ tiền xử lý ảnh
    logger.info(f"Đang lưu mô hình vào thư mục: {config.MODEL_DIR}...")
    model.save_pretrained(config.MODEL_DIR)
    hf_image_processor.save_pretrained(config.MODEL_DIR)
    logger.info("Lưu mô hình thành công!")
    
    # Vẽ và lưu biểu đồ Loss & Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.EPOCHS + 1), epoch_losses, 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.EPOCHS + 1), epoch_accuracies, 'g-o')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    chart_path = config.OUTPUT_DIR / "training_curves.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    logger.info(f"Biểu đồ huấn luyện đã lưu tại: {chart_path}")

if __name__ == "__main__":
    main()
