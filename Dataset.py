from datasets import load_dataset
import torch
import numpy as np
from datasets import load_from_disk
import torch.multiprocessing as mp
import psutil, os
from torch.utils.data import Dataset, DataLoader

def hf_to_tensor_split(split, out_prefix):
    images = []
    labels = []
    for example in split:
        img = torch.from_numpy(np.array(example["image"], dtype=np.float32)).permute(2,0,1) / 255.0
        label = torch.tensor(example["label"], dtype=torch.long)
        images.append(img)
        labels.append(label)

    images = torch.stack(images)   # [N,C,H,W]
    labels = torch.stack(labels)   # [N]
    torch.save(images, f"{out_prefix}_images.pt")
    torch.save(labels, f"{out_prefix}_labels.pt")
    print(f"Saved {out_prefix}: {images.shape}, {labels.shape}")

def split_to_tensors(split):
    imgs, labs = [], []
    for ex in split: 
        img = torch.from_numpy(np.array(ex["image"], dtype=np.float32)).permute(2,0,1) / 255.0  # [C,H,W]
        lab = torch.tensor(ex["label"], dtype=torch.long)
        imgs.append(img); labs.append(lab)
    images = torch.stack(imgs)   # [N,C,H,W]
    labels = torch.stack(labs)   # [N]
    return images, labels

class SharedTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return self.labels.shape[0]
    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform: x = self.transform(x)
        return x, y

def tensor_info(name, t):
    mem_mb = t.numel() * t.element_size() / 1024 / 1024
    return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"numel={t.numel()}, size={mem_mb:.2f} MB, data_ptr={t.storage().data_ptr()}")

def worker(rank, images, labels):
    pid = os.getpid()
    print(f"\n[Rank {rank}] PID={pid} started")
    print(tensor_info(f"[Rank {rank}] images", images))
    print(tensor_info(f"[Rank {rank}] labels", labels))

    rss = psutil.Process(pid).memory_info().rss / 1024 / 1024
    print(f"[Rank {rank}] RSS memory usage: {rss:.2f} MB")
    print(f"[Rank {rank}] done.\n")
    dataset = SharedTensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    while True: 
        for i, (img, lab) in enumerate(dataloader):
            print(f"[Rank {rank}] working with images {i} shape {img.shape}...")
            import time; time.sleep(5)

def get_dataset(ds_config):
    if ds_config.name == "celeba_hq":
        # train_imgs = torch.load("./data/celeba-hq-256x256/train_images.pt")
        # train_labs = torch.load("./data/celeba-hq-256x256/train_labels.pt")
        val_imgs = torch.load("./data/celeba-hq-256x256/val_images.pt")
        val_labs = torch.load("./data/celeba-hq-256x256/val_labels.pt")
        # train_dataset = SharedTensorDataset(train_imgs, train_labs)
        val_dataset = SharedTensorDataset(val_imgs, val_labs)
        return 1, val_dataset
    elif ds_config.name == "ariamp4":
        val_imgs = torch.load("./data/aria_256x256/dataset.pt")
        val_labs = torch.zeros(val_imgs.shape[0], dtype=torch.long)  # dummy labels
        val_dataset = SharedTensorDataset(val_imgs, val_labs)
        return 1, val_dataset
    else:
        raise ValueError(f"Unsupported dataset: {ds_config.name}")


if __name__ == "__main__":
    
    # Prepare dataset if not exist
    if os.path.exists("./data/celeba-hq-256x256/train_images.pt"):
        print("Dataset already prepared.")
    else:
        ds = load_dataset("korexyz/celeba-hq-256x256", cache_dir="./cache_celeba")
        os.makedirs("./data/celeba-hq-256x256", exist_ok=True)
        hf_to_tensor_split(ds["train"], "./data/celeba-hq-256x256/train")
        hf_to_tensor_split(ds["validation"], "./data/celeba-hq-256x256/val")

    
    mp.set_start_method("spawn", force=True)
    
    train_imgs = torch.load("./data/celeba-hq-256x256/train_images.pt")
    train_labs = torch.load("./data/celeba-hq-256x256/train_labels.pt")
    # val_imgs = torch.load("./data/celeba-hq-256x256/val_images.pt")
    # val_labs = torch.load("./data/celeba-hq-256x256/val_labels.pt")
    
    train_imgs.share_memory_()
    train_labs.share_memory_()
    # val_imgs.share_memory_()
    # val_labs.share_memory_()

    print("[Main] in parent process")
    print(tensor_info("[Main] images", train_imgs))
    print(tensor_info("[Main] labels", train_labs))
    rss_main = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[Main] RSS memory usage: {rss_main:.2f} MB\n")

    mp.spawn(worker, args=(train_imgs, train_labs), nprocs=16, join=True)