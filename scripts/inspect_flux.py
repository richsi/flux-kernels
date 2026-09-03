# scripts/inspect_flux.py
import torch
from diffusers import FluxTransformer2DModel

print("Instantiating FLUX architecture on meta device...")

# Build ONLY the FLUX Transformer in-memory (0 VRAM, no network calls)
with torch.device("meta"):
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        num_layers=19,               # 19 Double Stream Blocks
        num_single_layers=38,        # 38 Single Stream Blocks
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        axes_dims_rope=(16, 56, 56),
    )

print("\n=== TOP LEVEL SUBMODULES ===")
for name, child in model.named_children():
    print(f"- {name}: {child.__class__.__name__}")

print("\n=== DOUBLE BLOCK [0] STRUCTURE ===")
print(model.transformer_blocks[0])

print("\n=== SINGLE BLOCK [0] STRUCTURE ===")
print(model.single_transformer_blocks[0])