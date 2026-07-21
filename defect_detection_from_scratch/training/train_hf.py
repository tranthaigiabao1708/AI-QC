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

# Thêm đường dẫn gốc vào python path để import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from dataset import DefectDataset

def get_image_files(directory):
    """Lấy danh sách tất cả các đường dẫn ảnh trong thư mục."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    directory = Path(directory)
    if not directory.exists():
        return []
    return [f for f in directory.iterdir() if f.suffix.lower() in exts]

def main():
    logger.info("=== BẮT ĐẦU THỰC HÀNH HUẤN LUYỆN MODEL HUGGING FACE ===")
    
    # 1. Thu thập dữ liệu
    ok_dir = config.DATA_DIR / "OK"
    ng_dir = config.DATA_DIR / "NG"
    
    ok_files = get_image_files(ok_dir)
    ng_files = get_image_files(ng_dir)
    
    logger.info(f"Tìm thấy {len(ok_files)} ảnh OK và {len(ng_files)} ảnh NG")
    
    if len(ok_files) == 0 or len(ng_files) == 0:
        logger.error(f"Thiếu dữ liệu huấn luyện! Đặt ảnh mẫu vào thư mục OK và NG.")
        sys.exit(1)
        
    file_paths = ok_files + ng_files
    labels = [0] * len(ok_files) + [1] * len(ng_files)
    
    # =====================================================================
    # TODO: BƯỚC 1: TẢI BỘ VI XỬ LÝ ẢNH & MÔ HÌNH HUGGING FACE
    # =====================================================================
    logger.info(f"Đang tải Hugging Face Image Processor...")
    # Gợi ý: Dùng AutoImageProcessor.from_pretrained(config.MODEL_NAME)
    hf_image_processor = None # TODO: Điền code tải ImageProcessor
    
    logger.info(f"Đang tải mô hình pre-trained: {config.MODEL_NAME}...")
    # Gợi ý:
    # model = AutoModelForImageClassification.from_pretrained(
    #     config.MODEL_NAME,
    #     num_labels=len(config.CLASS_NAMES),
    #     id2label={i: c for i, c in enumerate(config.CLASS_NAMES)},
    #     label2id={c: i for i, c in enumerate(config.CLASS_NAMES)},
    #     ignore_mismatched_sizes=True
    # )
    model = None # TODO: Điền code tải Model
    
    # =====================================================================
    # TODO: BƯỚC 2: THIẾT LẬP TRANSFER LEARNING (ĐÓNG BĂNG BACKBONE)
    # =====================================================================
    logger.info("Thiết lập Transfer Learning...")
    # Gợi ý: 
    # 1. Chạy vòng lặp qua model.parameters() và set requires_grad = False
    # 2. Tìm lớp classifier cuối cùng (thường là model.classifier) và set requires_grad = True cho các parameter của nó.
    
    # TODO: Viết code đóng băng toàn bộ trọng số của mô hình
    
    # TODO: Viết code mở khóa huấn luyện cho lớp phân loại cuối cùng (classifier)
    
    # Di chuyển mô hình lên thiết bị tính toán (GPU hoặc CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model.to(device)
    logger.info(f"Huấn luyện trên thiết bị: {device}")
    
    # =====================================================================
    # TODO: BƯỚC 3: THIẾT LẬP DATASET & DATALOADER PYTORCH
    # =====================================================================
    # Tạo Dataset thực hành với augmentation = True
    train_dataset = DefectDataset(
        image_paths=file_paths,
        labels=labels,
        hf_image_processor=hf_image_processor,
        augment=True,
        n_augment=config.N_AUGMENT
    )
    
    eval_dataset = DefectDataset(
        image_paths=file_paths,
        labels=labels,
        hf_image_processor=hf_image_processor,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # =====================================================================
    # TODO: BƯỚC 4: THIẾT LẬP OPTIMIZER & LOSS FUNCTION
    # =====================================================================
    # Gợi ý:
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE)
    # criterion = nn.CrossEntropyLoss()
    optimizer = None # TODO: Tạo bộ tối ưu hóa AdamW
    criterion = None # TODO: Tạo hàm tính loss CrossEntropyLoss
    
    # =====================================================================
    # TODO: BƯỚC 5: VIẾT VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
    # =====================================================================
    logger.info("Bắt đầu chạy vòng lặp huấn luyện...")
    epoch_losses = []
    epoch_accuracies = []
    
    if model is None or optimizer is None or criterion is None:
        logger.error("Vui lòng hoàn thiện TODO Bước 1, 2, 4 trước khi chạy huấn luyện!")
        sys.exit(1)
        
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].to(device)
            
            # TODO: 1. Reset gradient bằng optimizer.zero_grad()
            
            # TODO: 2. Forward pass: Tính đầu ra mô hình bằng model(pixel_values=pixel_values)
            # outputs = ...
            # logits = outputs.logits
            
            # TODO: 3. Tính Loss: loss = criterion(logits, batch_labels)
            
            # TODO: 4. Backward pass: loss.backward() để tính đạo hàm ngược
            
            # TODO: 5. Cập nhật trọng số: optimizer.step()
            
            # TODO: 6. Cộng dồn running_loss và tính correct_preds để theo dõi accuracy
            
            pass # Xóa dòng pass này khi viết code
            
        # Tính Loss & Accuracy trung bình của Epoch
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        
        logger.info(f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1%}")
        
    logger.info("Hoàn tất huấn luyện!")
    
    # =====================================================================
    # TODO: BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH VÀ LƯU CHECKPOINT
    # =====================================================================
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].to(device)
            
            # TODO: Thực hiện forward pass và lấy nhãn có xác suất cao nhất
            
            pass # Xóa dòng pass này khi viết code
            
    # Tính các chỉ số đánh giá
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=config.CLASS_NAMES, zero_division=0)
    
    logger.info(f"Độ chính xác trên tập dữ liệu gốc: {acc:.1%}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    
    # TODO: Lưu mô hình và bộ tiền xử lý ảnh sử dụng save_pretrained()
    # Gợi ý:
    # model.save_pretrained(config.MODEL_DIR)
    # hf_image_processor.save_pretrained(config.MODEL_DIR)
    logger.info(f"Lưu mô hình thành công!")
    
    # Vẽ và lưu biểu đồ Loss & Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.EPOCHS + 1), epoch_losses, 'b-o')
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.EPOCHS + 1), epoch_accuracies, 'g-o')
    plt.title('Training Accuracy')
    
    chart_path = config.OUTPUT_DIR / "training_curves.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    logger.info(f"Biểu đồ đã lưu tại: {chart_path}")

if __name__ == "__main__":
    main()
