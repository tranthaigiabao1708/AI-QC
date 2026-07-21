import os
import sys
import io

# Fix Unicode output trên Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import torch
from transformers import AutoModelForImageClassification
from loguru import logger

# Thêm đường dẫn gốc vào python path để import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def export_to_onnx():
    logger.info("=== BẮT ĐẦU XUẤT MÔ HÌNH SANG ĐỊNH DẠNG ONNX ===")
    
    # 1. Đường dẫn mô hình
    model_dir = config.MODEL_DIR
    onnx_path = config.OUTPUT_DIR / "model.onnx"
    
    if not (model_dir / "config.json").exists():
        logger.error(f"Không tìm thấy mô hình PyTorch đã huấn luyện tại: {model_dir}")
        logger.error("Vui lòng chạy huấn luyện model bằng 'train_hf.py' trước khi export.")
        return
        
    # 2. Tải mô hình
    logger.info(f"Đang tải mô hình PyTorch từ: {model_dir}...")
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    
    # 3. Tạo một tensor đầu vào giả (Dummy Input) để định hình mạng nơ-ron
    # Batch size = 1, Channels = 3 (RGB), Height = 224, Width = 224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Thực hiện export sang ONNX
    logger.info(f"Đang biên dịch và xuất sang file ONNX tại: {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,        # Lưu cả trọng số (weights) đã học
            opset_version=14,           # Phiên bản ONNX opset khuyên dùng
            do_constant_folding=True,   # Tối ưu hóa hằng số trong đồ thị tính toán
            input_names=["pixel_values"], # Tên đầu vào của mô hình
            output_names=["logits"],     # Tên đầu ra của mô hình
            dynamic_axes={              # Cho phép thay đổi Batch size động khi suy luận
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"}
            }
        )
        logger.info("Xuất mô hình ONNX thành công!")
        
        # 5. So sánh kích thước file
        pytorch_size = sum(p.stat().st_size for p in model_dir.glob("*.bin")) / (1024 * 1024)
        if pytorch_size == 0:
            # Hugging face mới có thể lưu dạng safetensors
            pytorch_size = sum(p.stat().st_size for p in model_dir.glob("*.safetensors")) / (1024 * 1024)
            
        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"Kích thước PyTorch gốc: {pytorch_size:.2f} MB")
        logger.info(f"Kích thước ONNX xuất ra:  {onnx_size:.2f} MB")
        logger.info("---")
        logger.info("💡 Ý NGHĨA AI ENGINEER:")
        logger.info("ONNX giúp mô hình chạy suy luận độc lập không cần thư viện PyTorch/Transformers.")
        logger.info("Bạn có thể load file model.onnx này bằng thư viện 'onnxruntime' siêu nhẹ để deploy")
        logger.info("trên các thiết bị nhúng (Jetson, Raspberry Pi) hoặc C++ với hiệu năng cực cao.")
        
    except Exception as e:
        logger.error(f"Lỗi xuất ONNX: {e}")

if __name__ == "__main__":
    export_to_onnx()
