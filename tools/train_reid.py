#!/usr/bin/env python3
"""Train the Re-ID network (paper Sec. IV-A).

ResNet50, softmax classification loss, crops of the same physical lumen form
one class, crops resized to 128x128. At inference the FC layer is dropped.

Dataset layout (ImageFolder): root/<lumen_id>/*.png  -- tools/make_reid_crops.py
builds it from labelled frames.

python tools/train_reid.py --data reid_crops --out reid_resnet50.pth --epochs 60
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    import torch
    import torchvision
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reid_resnet50.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--no-imagenet", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = T.Compose([T.Resize((128, 128)), T.RandomRotation(180), T.ColorJitter(0.2, 0.2, 0.1),
                          T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_val = T.Compose([T.Resize((128, 128)), T.ToTensor(), norm])
    full = torchvision.datasets.ImageFolder(a.data, transform=tf_train)
    n_val = int(len(full) * a.val)
    tr, va = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    va.dataset = torchvision.datasets.ImageFolder(a.data, transform=tf_val)
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    dv = DataLoader(va, a.bs, num_workers=a.workers)
    print(f"{len(full)} crops, {len(full.classes)} lumen identities")

    w = None if a.no_imagenet else torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights=w)
    net.fc = torch.nn.Linear(2048, len(full.classes))
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    for ep in range(a.epochs):
        net.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = ce(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sch.step()
        net.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in dv:
                p = net(x.to(dev)).argmax(1).cpu()
                correct += (p == y).sum().item()
                n += len(y)
        acc = correct / max(n, 1)
        print(f"epoch {ep + 1:3d}  loss {tot / len(tr):.4f}  val acc {acc:.3f}")
        if acc >= best:
            best = acc
            torch.save(net.state_dict(), a.out)
    print(f"best val acc {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
