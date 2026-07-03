import torch
from diffusers.utils import load_image, check_min_version
from diffusers import DiffusionPipeline, DDIMScheduler
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from submodules.flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
from submodules.flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel
from submodules.flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor, gaussian_blur
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
from simulation.utils import dilate_binary_mask, smooth_segmentation_mask_255
import sys
import os
sys.path.append(os.path.abspath("submodules/flux_controlnet_inpainting"))

check_min_version("0.30.2")

class FluxInpainter:
    # def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None
        self.load_model()
        # self.load_model_attneraser()
        
    def load_model(self):
        """Load the FLUX ControlNet inpainting model and pipeline"""
        # Load ControlNet
        # controlnet = FluxControlNetModel.from_pretrained(
        #     "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", 
        #     torch_dtype=self.torch_dtype
        # )
        
        # # Load Transformer
        # transformer = FluxTransformer2DModel.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev", 
        #     subfolder='transformer', 
        #     torch_dtype=self.torch_dtype
        # )
        
        # # Build pipeline
        # self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev",
        #     controlnet=controlnet,
        #     transformer=transformer,
        #     torch_dtype=self.torch_dtype
        # ).to(self.device)
        
        scheduler = DDIMScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False
        )
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            custom_pipeline="/home/huteng/lliqirui/AttentionEraser/pipeline_stable_diffusion_xl_attentive_eraser.py",
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        
        # Ensure components are in correct dtype
        # self.pipe.transformer.to(self.torch_dtype)
        # self.pipe.controlnet.to(self.torch_dtype)
        
        print("Model loaded successfully")
        
    def __call__(self, image, mask, prompt="", size=(512, 512), 
                      num_inference_steps=24, controlnet_conditioning_scale=0.9,
                      guidance_scale=3.5, negative_prompt="", true_guidance_scale=3.5,
                      seed=42):
        """Run inpainting with the given parameters"""
        if self.pipe is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        def preprocess_image(image, device):
            # Ensure image is 4D: (N, C, H, W)
            if image.dim() == 3:  # (C, H, W)
                image = image.unsqueeze(0)
            elif image.dim() == 2:  # (H, W) grayscale
                image = image.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
            # Now image is 4D
            image = image.float() * 2 - 1  # [0,1] --> [-1,1]
            if image.shape[1] != 3:
                image = image.expand(-1, 3, -1, -1)
            image = F.interpolate(image, (1024, 1024))
            image = image.to(self.torch_dtype).to(device)
            return image

        def preprocess_mask(mask, device):
            # Ensure mask is 4D: (N, C, H, W)
            if mask.dim() == 2:  # (H, W)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:  # (C, H, W) or (1, H, W)
                mask = mask.unsqueeze(0)
            # Now mask is 4D
            mask = mask.float()  # 0 or 1
            mask = F.interpolate(mask, (1024, 1024))
            mask = gaussian_blur(mask, kernel_size=(77, 77))
            mask[mask < 0.1] = 0
            mask[mask >= 0.1] = 1
            mask = mask.to(self.torch_dtype).to(device)
            return mask


        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = preprocess_image(image, self.device)
        mask = preprocess_mask(mask, self.device)

        result = self.pipe(
            prompt="", # Attentive Eraser 不需要 prompt 即可工作
            image=image,
            mask_image=mask,
            height=1024,
            width=1024,
            AAS=True, 
            strength=0.8, 
            rm_guidance_scale=9, 
            ss_steps=9, 
            ss_scale=0.3, 
            AAS_start_step=0, 
            AAS_start_layer=34, 
            AAS_end_layer=70, 
            num_inference_steps=50, 
            generator=generator,
            guidance_scale=1,
        ).images[0]  # Returns PIL Image
        # Resize to expected size if needed
        print(f'size: {size[0],size[1]}')
        result = result.resize((size[0], size[1]), Image.LANCZOS)
        return result
