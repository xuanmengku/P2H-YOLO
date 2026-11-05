import matplotlib

matplotlib.use("Agg")

from ultralytics import YOLO


def main():
    model = YOLO("yolo26m-p2h.yaml")  # Select the model configuration YAML

    model.train(
        data="dataset_configs/VisDrone2019.yaml",
        epochs=150,
        batch=4,
        imgsz=640,
        device="1",
        workers=4,
        optimizer="MuSGD",
        lr0=0.01,
        pretrained=False,
        augment=True,
        patience=50,
        seed=42,
        amp=True,
        cache=False,
        rect=False,
        cos_lr=True,
        label_smoothing=0.0,
        dropout=0.0,
        verbose=True,
        project="runs/train",
        name="visdrone2019",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
