# pip install opencv-python torch
import os, glob, cv2, torch, numpy as np

# Folder containing mp4 files
video_dir         = "./data/aria_videos"   # mp4 folder
output_pt         = "./data/aria_256x256"             
frames_per_video  = 10                        
target_size       = 256                         
os.makedirs(output_pt, exist_ok=True)
# ========================

def sample_indices(n_total, k):
    k = min(k, max(int(n_total), 0))
    if k <= 0: return []
    if k == 1: return [0]
    return sorted(set(np.linspace(0, n_total - 1, k, dtype=int).tolist()))

all_imgs, meta = [], []   
mp4s = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
print(f"Found {len(mp4s)} mp4")

for vid in mp4s:
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"[skip] cannot open: {vid}")
        continue
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(n, frames_per_video)

    for fidx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # BGR->RGB, resize -> 256x256
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        # HWC(uint8) -> CHW(uint8)
        img = torch.from_numpy(frame)              # [H,W,C], uint8
        img = img.permute(2, 0, 1).contiguous()    # [C,H,W]
        all_imgs.append(img)
        meta.append((vid, int(fidx)))
    cap.release()

if not all_imgs:
    raise RuntimeError("No frames collected. Check folder path or videos.")

data = torch.stack(all_imgs, dim=0)   # [B,C,256,256], uint8

# [B,C,256,256], uint8  -> float32 in [0,1]
data = torch.stack(all_imgs, dim=0).float().div_(255.0)   # [B,C,256,256], float32
print("Final tensor:", data.shape, data.dtype, data.min().item(), data.max().item())

# Save to float32 tensor
torch.save(data, os.path.join(output_pt, "dataset.pt"))
print(f"Saved to {output_pt}")


import matplotlib.pyplot as plt

save_dir = "samples"
save_dir = os.path.join(output_pt, save_dir)
k = 6
os.makedirs(save_dir, exist_ok=True)

B = data.shape[0]
idxs = np.linspace(0, B - 1, num=min(k, B), dtype=int)

for j, i in enumerate(idxs):
    img = data[i].permute(1, 2, 0).cpu().numpy()  # float32, [0,1], [H,W,C]
    plt.imsave(os.path.join(save_dir, f"sample_{j:03d}.png"), img)  # Range in [0,1]


