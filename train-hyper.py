# train.py
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO

def main():
    # 1. 加载预训练模型（自动从官网下载或加载本地缓存）
    # 可选模型: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO("yolo26m-drone-hyper.yaml")  # 使用nano版本预训练模型

    # 2. 训练模型
    results = model.train(
        data="/mnt/disk1/ultralytics-wzc/VisDrone/data.yaml",  # 数据集配置文件路径
        epochs=150,                 # 训练轮次
        batch=4,                   # 批次大小（根据GPU调整）
        imgsz=640,                  # 输入图像尺寸
        device="1",                 # 使用GPU 0（"cpu"表示使用CPU）
        workers=4,                  # 数据加载线程数
        optimizer="auto",           # 优化器（推荐自动选择）
        lr0=0.01,
        #name="exp",        # 实验名称（结果保存在runs/detect/custom_train）
        pretrained=False,           # 使用预训练权重
        augment=True,              # 启用数据增强
        patience=50,               # 早停轮次
        seed=42,                   # 随机种子
        # ----- 显存优化关键参数 -----
        amp=True,                  # 混合精度训练（大幅节省显存，推荐）
        cache=False,               # 不缓存图像（若数据集不大可改为 True 加速，但会占用更多内存）
        rect=False,                # 矩形训练（可略微节省显存，但会改变 batch 形状）
        cos_lr=True,               # 余弦学习率调度（提升收敛）
        label_smoothing=0.0,       # 标签平滑（可尝试 0.05）
        dropout=0.0,               # 若模型本身无 dropout 则不用
        verbose=True,              # 打印详细日志  
        #
        project="/mnt/disk1/lixue/out",
        name="yolo/hyper/visdrone2019/train3_1",
        exist_ok=True,         
    )

if __name__ == "__main__":
    main()