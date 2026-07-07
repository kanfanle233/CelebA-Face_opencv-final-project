# train.py
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from celeba_utils import (
    MODEL_PATH,
    TRAIN_RESULTS_DIR,
    load_celeba_full_df,
    load_image,
    align_face_by_eyes,
)


# ========== 2. Dataset 类定义（放在全局没问题） ==========
class CelebAFaceDataset(Dataset):
    def __init__(self, df, target_attrs, image_size=(128, 128), augment=False):
        self.df = df.reset_index(drop=True)
        self.target_attrs = target_attrs
        self.image_size = image_size

        base_tf = [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std =[0.5, 0.5, 0.5]),
        ]
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                *base_tf
            ])
        else:
            self.transform = T.Compose(base_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_raw = load_image(row)  # BGR
        face = align_face_by_eyes(img_raw, row, output_size=self.image_size)  # BGR

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(face_rgb)
        x = self.transform(img_pil)  # [3,H,W]

        y_vals = row[self.target_attrs].values.astype(np.float32)
        y = torch.from_numpy(y_vals)  # [num_classes]
        return x, y


# ========== 3. 小型 CNN 定义（也可以全局放） ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # 3x128x128 -> 32x64x64
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32x64x64 -> 64x32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits


# ========== 主逻辑封装到 main() 里，避免 Windows 多进程报错 ==========
def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 0. 设备
    assert torch.cuda.is_available(), "没有检测到 CUDA，请检查 RTX 4060 驱动"
    device = torch.device("cuda:0")
    print("当前 GPU:", torch.cuda.get_device_name(device))

    # 1. 加载 full_df
    full_df = load_celeba_full_df()
    print("CelebA 总样本数:", len(full_df))

    # 选择属性：性别 + 眼镜
    target_attrs = ["Male", "Eyeglasses"]
    num_classes = len(target_attrs)

    # -1 / 1 → 0 / 1
    for col in target_attrs:
        full_df[col] = full_df[col].replace({-1: 0, 1: 1})

    # 按 split 划分 train/val（0=train, 1=val）
    train_df = full_df[full_df["split"] == 0].reset_index(drop=True)
    val_df   = full_df[full_df["split"] == 1].reset_index(drop=True)
    print("Train size:", len(train_df))
    print("Val size  :", len(val_df))

    # 2. Dataset & DataLoader
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 512
    NUM_WORKERS = 4  # Windows 下先用 0，想提速再改 4

    train_dataset = CelebAFaceDataset(train_df, target_attrs, IMAGE_SIZE, augment=True)
    val_dataset   = CelebAFaceDataset(val_df,   target_attrs, IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. 模型
    model = SimpleCNN(num_classes).to(device)
    print(model)
    print("可训练参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 4. 损失、优化器
    criterion = nn.BCEWithLogitsLoss()
    # 默认学习率
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 5. 验证函数（跟之前一样）
    def evaluate(model, loader):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_per_attr = torch.zeros(num_classes, device=device)

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct_per_attr += (preds == yb).sum(dim=0)

        avg_loss = total_loss / total_samples
        acc_per_attr = (correct_per_attr / total_samples).cpu().numpy()
        return avg_loss, acc_per_attr

    # 6. 如果存在 best_model.pth，就在它基础上继续训
    best_acc = 0.0
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"检测到 {MODEL_PATH}，已加载权重，在此基础上继续训练。")

        # 可选：继续训练时把学习率稍微调低一点（比如 /2）
        for g in optimizer.param_groups:
            g["lr"] = lr * 0.5
        print(f"继续训练，学习率调整为 {lr * 0.5:g}")

        # 先在 val 上评估一次，作为当前 best_acc
        val_loss0, val_acc0 = evaluate(model, val_loader)
        best_acc = float(val_acc0.mean())
        print(f"当前加载模型在验证集上的平均准确率: {best_acc:.4f} (val_loss={val_loss0:.4f})")
    else:
        print("未找到 best_model.pth，将从随机初始化开始训练。")

    # 7. 训练循环
    EPOCHS = 32

    history = {
        "train_loss": [],
        "val_loss": [],
        "acc": {name: [] for name in target_attrs}
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            mem = torch.cuda.memory_allocated(device) / 1024**3
            pbar.set_postfix(loss=running_loss/total, mem=f"{mem:.2f}GB")

        scheduler.step()

        train_loss = running_loss / total
        val_loss, val_acc_attr = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for name, acc in zip(target_attrs, val_acc_attr):
            history["acc"][name].append(acc)

        avg_acc = float(val_acc_attr.mean())
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"更新并保存新的 {MODEL_PATH}, 平均 acc = {best_acc:.4f}")

        acc_str = ", ".join([f"{n}={a:.3f}" for n, a in zip(target_attrs, val_acc_attr)])
        print(f"\n[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  |  {acc_str}\n")

    print("继续训练结束。")
    print("当前最佳验证平均准确率:", best_acc)

    # ==== 8. 画损失曲线 & 准确率曲线 ====
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = TRAIN_RESULTS_DIR / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.show()

    # 每个属性的准确率曲线
    plt.figure(figsize=(6, 4))
    for name, vals in history["acc"].items():
        plt.plot(epochs_range, vals, label=f"{name} Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Attribute")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    acc_curve_path = TRAIN_RESULTS_DIR / "acc_curve.png"
    plt.savefig(acc_curve_path, dpi=200)
    plt.show()

    print(f"损失曲线已保存为 {loss_curve_path}，准确率曲线已保存为 {acc_curve_path}")




# ========== Windows 多进程 DataLoader 必备入口 ==========
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 可能已经设置过 start_method，忽略即可
        pass

    main()
