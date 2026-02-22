# i dont have generative folder, i just generated one image, from below code, can you generate multiple using below code: 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random

device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------------
# UNET (same architecture as training)
# -------------------------------------------------------
class GeologyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._block(1, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.bottleneck = self._block(512, 512)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )

        self.dec1 = self._block(512 + 512, 256)
        self.dec2 = self._block(256 + 256, 128)
        self.dec3 = self._block(128 + 128, 64)
        self.dec4 = self._block(64 + 64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))

        bott = self.bottleneck(self.down(e4))
        bott = bott + t_emb

        d1 = self.dec1(torch.cat([self.up(bott), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up(d3), e1], dim=1))

        return self.final(d4)


# -------------------------------------------------------
# Load trained model
# -------------------------------------------------------
model = GeologyUNet().to(device)
model.load_state_dict(torch.load("/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/train_geology_output/model100.pth",map_location=device))
model.eval()

# -------------------------------------------------------
# DDPM parameters (same as training)
# -------------------------------------------------------
T = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# -------------------------------------------------------
# Dataset loader (normalized to [-1,1])
# -------------------------------------------------------
class GeologyNPY(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.float32)    # (64,256)

        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)   # [0,1]
        arr = arr * 2 - 1                                           # [-1,1]

        return torch.from_numpy(arr).unsqueeze(0), 0


dataset = GeologyNPY("/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/slice_XZ_numpy_patches")
# -----------------------------
# Check training data mean
# -----------------------------
train_mean = []
for i in range(100):
    arr, _ = dataset[random.randint(0, len(dataset)-1)]
    train_mean.append(arr.mean().item())

print("Average training mean:", np.mean(train_mean))

# pick random slices
B = 4
idx = random.sample(range(len(dataset)), B)
image = torch.stack([dataset[i][0] for i in idx]).to(device)

print("Selected slices:", idx)

# -------------------------------------------------------
# Mask (right-side removed)
# -------------------------------------------------------
mask = torch.ones_like(image)
mask[:, :, :, 120:] = 0
known_region = image * mask


# -------------------------------------------------------
# RePaint (correct for [-1,1])
# -------------------------------------------------------
def repaint(model, x0, mask, T, jump_length, jump_n_sample):
    model.eval()
    device = x0.device
    B = x0.size(0)

    x_t = torch.randn_like(x0)

    for t in tqdm(range(T - 1, -1, -1)):

        for u in range(jump_n_sample if (t > 0 and t % jump_length == 0) else 1):

            t_tensor = torch.tensor([t] * B, device=device).long()

            with torch.no_grad():
                pred_noise = model(x_t, t_tensor)

            # ---- known region sample at t-1
            if t > 0:
                noise_k = torch.randn_like(x0)
                x_known = (
                    sqrt_alphas_cumprod[t-1] * x0 +
                    sqrt_one_minus_alphas_cumprod[t-1] * noise_k
                )
            else:
                x_known = x0

            # ---- unknown region DDPM step
            alpha_t = alphas[t]
            alpha_bar = alphas_cumprod[t]

            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar)) * pred_noise
            )

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(betas[t])
            else:
                noise = 0

            x_unknown = mean + sigma * noise

            # ---- merge known & unknown
            x_t_minus_1 = mask * x_known + (1 - mask) * x_unknown

            # ---- jump sampling
            if u < jump_n_sample - 1 and t > 0:
                noise_r = torch.randn_like(x_t_minus_1)
                x_t = (
                    torch.sqrt(alphas[t-1]) * x_t_minus_1 +
                    torch.sqrt(1 - alphas[t-1]) * noise_r
                )
            else:
                x_t = x_t_minus_1
    output = x_t
    
    return output  # stays in [-1,1]


# -------------------------------------------------------
# Unconditional DDPM Sampling
# -------------------------------------------------------

output = repaint(model, known_region, mask, T, jump_length=5, jump_n_sample=20)



# Add this to your RePaint script to generate multiple samples:

# def generate_multiple_repaint_samples(model, dataset, num_samples=10, 
#                                      mask_position=120, save_dir='/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/repaint_generated'):
#     """Generate multiple inpainting samples"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Select random samples from dataset
#     indices = random.sample(range(len(dataset)), num_samples)
#     all_outputs = []
    
#     for i, idx in enumerate(indices):
#         print(f"Processing sample {i+1}/{num_samples}")
        
#         # Load single image
#         image = dataset[idx][0].unsqueeze(0).to(device)  # [1, 1, 64, 256]
        
#         # Create mask
#         mask = torch.ones_like(image)
#         mask[:, :, :, mask_position:] = 0
#         known_region = image * mask
        
#         # Run RePaint
#         output = repaint(model, known_region, mask, T, jump_length=5, jump_n_sample=20)
        
#         # Convert to [0, 1]
#         # output_norm = (output.detach().cpu() + 1) / 2
#         output_norm = output.detach().cpu()
#         # output_norm = torch.clamp(output_norm, 0, 1)
        
#         # Save PNG
#         plt.figure(figsize=(12, 4))
        
#         # Original
#         plt.subplot(1, 3, 1)
#         orig_vis = image[0,0].cpu()
#         plt.imshow(orig_vis, cmap="gray", vmin=-1, vmax=1)
#         plt.title("Original")
        

#         # Masked
#         plt.subplot(1, 3, 2)
#         plt.imshow((image*mask)[0,0].cpu(), cmap="gray", vmin=-1, vmax=1)
#         H, W = (image*mask)[0,0].cpu().shape
#         pink_overlay = np.zeros((H, W, 4))         #combining red and blue makes pink/magenta
#         pink_overlay[:, mask_position:, 0] = 1.0   # this is red
#         pink_overlay[:, mask_position:, 2] = 1.0   # ths is blue
#         pink_overlay[:, mask_position:, 3] = 1.0   #to remove transparency

#         plt.imshow(pink_overlay)
#         plt.title("Masked")

#         # RePainted
#         plt.subplot(1, 3, 3)
#         repaint_vis = output_norm[0, 0]
#         plt.imshow(repaint_vis, cmap="gray", vmin=-1, vmax=1)
#         plt.title("RePainted")

        
#         plt.tight_layout()
#         plt.savefig(f'{save_dir}/repaint_sample_{i:03d}.png', dpi=150, bbox_inches='tight')
#         plt.close()
        
#         # Save numpy for evaluation
#         np.save(f'{save_dir}/repaint_sample_{i:03d}.npy', output_norm.numpy())
        
#         all_outputs.append(output_norm)
    
#     print(f"Generated {num_samples} inpainting samples in '{save_dir}' folder")
#     return all_outputs


def sample_unconditional(model, image_shape, T):
    model.eval()
    device = next(model.parameters()).device

    x_t = torch.randn(image_shape, device=device)

    for t in tqdm(range(T - 1, -1, -1)):
        t_tensor = torch.full((image_shape[0],), t, device=device, dtype=torch.long)

        with torch.no_grad():
            pred_noise = model(x_t, t_tensor)

        alpha_t = alphas[t]
        alpha_bar = alphas_cumprod[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar)) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(betas[t])
        else:
            noise = 0

        x_t = mean + sigma * noise

    return x_t

def generate_unconditional_ensemble(
    model,
    dataset,
    n_samples=100,
    save_dir='/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/unconditional_ensemble'
):

    os.makedirs(save_dir, exist_ok=True)
    sample_img = dataset[0][0]              # (C, H, W)
    image_shape = (1,) + sample_img.shape   # (B, C, H, W)

    print("Using image shape:", image_shape)

    outputs = []

    for i in range(n_samples):
        print(f"Unconditional sample {i+1}/{n_samples}")

        sample = sample_unconditional(model, image_shape, T)
        sample_cpu = sample.detach().cpu()

        plt.figure(figsize=(6, 4))
        plt.imshow(sample_cpu[0, 0], cmap="gray", vmin=-1, vmax=1)
        plt.title(f"Unconditional Sample {i}")
        plt.savefig(f'{save_dir}/sample_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

        np.save(f"{save_dir}/sample_{i:03d}.npy", sample_cpu.numpy())
        outputs.append(sample_cpu)

    print(f"Generated {n_samples} unconditional samples.")
    return torch.cat(outputs, dim=0)

# Stochastic conditional generator + uncertainty quantification
def generate_ensemble_for_single_condition(
    model,
    dataset,
    n_realizations=100,
    mask_position=120,
    save_dir='/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/repaint_ensemble'
):

    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # FIX ONE IMAGE (important!)
    # -----------------------------
    idx = random.randint(0, len(dataset)-1)
    image = dataset[idx][0].unsqueeze(0).to(device)

    mask = torch.ones_like(image)
    mask[:, :, :, mask_position:] = 0

    print(f"Fixed conditioning slice index: {idx}")

    outputs = []

    for i in range(n_realizations):
        print(f"RePaint realization {i+1}/{n_realizations}")

        output = repaint(model, image, mask, T, jump_length=5, jump_n_sample=20)
        output_cpu = output.detach().cpu()
        

        np.save(f'{save_dir}/repaint_sample_{i:03d}.npy', output_cpu.numpy())
        outputs.append(output_cpu)

    print(f"Generated {n_realizations} conditional samples.")
    return torch.cat(outputs, dim=0), image.detach().cpu(), mask.detach().cpu()

if __name__ == "__main__":
    # generated_samples = generate_multiple_repaint_samples(model, dataset, num_samples=10, mask_position=120, save_dir='/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/repaint_generated')
#     ensemble, fixed_image, fixed_mask = generate_ensemble_for_single_condition(
#     model,
#     dataset,
#     n_realizations=100,
#     mask_position=120
# )
    uncond_ensemble = generate_unconditional_ensemble(
    model,
    dataset,
    n_samples=100
)

    # Compute statistics
    uncond_mean = uncond_ensemble.mean(dim=0, keepdim=True)
    uncond_std  = uncond_ensemble.std(dim=0, keepdim=True)

    print(f"Unconditional Mean range: [{uncond_mean.min():.3f}, {uncond_mean.max():.3f}]")
    print(f"Unconditional Std  range: [{uncond_std.min():.3f}, {uncond_std.max():.3f}]")


    plt.figure(figsize=(6, 4))
    plt.imshow(uncond_mean[0, 0], cmap="gray", vmin=-1, vmax=1)
    plt.title("Pixel-wise Mean (Expected Model)")
    plt.colorbar()
    plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaintUnco_ensemble_mean6.png",
        dpi=150
    )
    plt.show()

    plt.figure(figsize=(6,4))
    plt.imshow(uncond_std[0,0], cmap="gray")
    plt.title("Unconditional Std (Diffusion Only)")
    plt.colorbar()
    plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaintUnco_ensemble_std6.png",
        dpi=150
    )
    plt.show()

    

    # =========================
    # STEP 2: Compute statistics
    # =========================
    # mean_img = ensemble.mean(dim=0, keepdim=True) #average geology
    # std_img  = ensemble.std(dim=0, keepdim=True) #uncertainty map

    # print(f"Mean range: [{mean_img.min():.3f}, {mean_img.max():.3f}]")
    # print(f"Std  range: [{std_img.min():.3f}, {std_img.max():.3f}]")

    # =========================
    # STEP 3A: Sample realizations
    # =========================
    # plt.figure(figsize=(15, 4))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(ensemble[i, 0], cmap="gray", vmin=-1, vmax=1)
    #     plt.title(f"Real {i+1}")
    #     plt.axis("off")
    # plt.suptitle("Conditional RePaint Realizations")
    # plt.savefig(
    #     "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaint_ensemble_sample-real4.png",
    #     dpi=150
    # )
    # plt.show()

    # =========================
    # STEP 3B: Mean model
    # =========================
    # plt.figure(figsize=(6, 4))
    # plt.imshow(mean_img[0, 0], cmap="gray", vmin=-1, vmax=1)
    # plt.title("Pixel-wise Mean (Expected Model)")
    # plt.colorbar()
    # plt.savefig(
    #     "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaint_ensemble_mean4.png",
    #     dpi=150
    # )
    # plt.show()

    # =========================
    # STEP 3C: Uncertainty map
    # =========================
    # plt.figure(figsize=(6, 4))
    # plt.imshow(std_img[0, 0], cmap="gray")
    # plt.title("Pixel-wise Std (Uncertainty)")
    # plt.colorbar()
    # plt.savefig(
    #     "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaint_ensemble_std4.png",
    #     dpi=150
    # )
    # plt.show()

    # =========================
    # STEP 3D: Conditioning overlay
    # =========================
    # mask_cpu = fixed_mask
    # plt.figure(figsize=(6, 4))
    # plt.imshow(std_img[0, 0], cmap="gray")
    # plt.imshow(mask_cpu[0, 0], cmap="gray", alpha=0.3)
    # plt.title("Uncertainty with Conditioning Overlay")
    # plt.colorbar()
    # plt.savefig(
    #     "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaint_ensemble_conditionalOverlay4.png",
    #     dpi=150
    # )
    # plt.show()

# Quick histogram after generating samples
# if ensemble.numel() > 0:

#     # Normalize generated to [0,1]
#     gen_values = ensemble.flatten()
#     gen_values = (gen_values - gen_values.min()) / (gen_values.max() - gen_values.min() + 1e-8)
#     gen_values = gen_values.cpu().numpy()

#     train_dir = "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/slice_XZ_numpy_patches"
#     train_files = random.sample(os.listdir(train_dir), 10)

#     train_values = []
#     for f in train_files:
#         arr = np.load(os.path.join(train_dir, f))
#         arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
#         train_values.append(arr.flatten())

#     train_values = np.concatenate(train_values)

#     plt.figure(figsize=(10, 5))
#     plt.hist(train_values, bins=50, alpha=0.5, density=True, label="Training")
#     plt.hist(gen_values, bins=50, alpha=0.5, density=True, label="Generated")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Density")
#     plt.title("Training vs Generated Distribution")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig(
#         "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaint_ensemble_quick_histogram3.png",
#         dpi=150
#     )
#     plt.show()

if uncond_ensemble.numel() > 0:

    # Normalize generated to [0,1]
    gen_values = uncond_ensemble.flatten()
    gen_values = (gen_values - gen_values.min()) / (gen_values.max() - gen_values.min() + 1e-8)
    gen_values = gen_values.cpu().numpy()

    train_dir = "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/slice_XZ_numpy_patches"
    train_files = random.sample(os.listdir(train_dir), 10)

    train_values = []
    for f in train_files:
        arr = np.load(os.path.join(train_dir, f))
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        train_values.append(arr.flatten())

    train_values = np.concatenate(train_values)

    plt.figure(figsize=(10, 5))
    plt.hist(train_values, bins=50, alpha=0.5, density=True, label="Training")
    plt.hist(gen_values, bins=50, alpha=0.5, density=True, label="Generated")
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.title("Training vs Generated Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/repaintUnco_ensemble_quick_histogram6.png",
        dpi=150
    )
    plt.show()

