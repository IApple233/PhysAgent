import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForInpainting
from diffusers.utils import check_min_version
from PIL import Image
from torchvision.transforms.functional import gaussian_blur, to_pil_image


check_min_version("0.30.2")


class FluxInpainter:
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ):
        self.device = device
        self.torch_dtype = torch.float32 if device == "cpu" else torch_dtype
        self.model_name = model_name
        self.pipe = None
        self.load_model()

    def load_model(self):
        """Load the inpainting pipeline from Hugging Face cache or download it."""
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            variant="fp16" if self.torch_dtype == torch.float16 else None,
            use_safetensors=True,
        ).to(self.device)
        print(f"Inpainting model loaded: {self.model_name}")

    def _image_to_pil(self, image, size):
        if isinstance(image, Image.Image):
            return image.convert("RGB").resize(size, Image.LANCZOS)

        if not torch.is_tensor(image):
            raise TypeError(f"Unsupported image type: {type(image).__name__}")

        image = image.detach().cpu().float()
        if image.dim() == 4:
            image = image[0]
        if image.dim() == 2:
            image = image.unsqueeze(0).expand(3, -1, -1)
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        image = image[:3].clamp(0, 1)
        return to_pil_image(image).resize(size, Image.LANCZOS)

    def _mask_to_pil(self, mask, size):
        if isinstance(mask, Image.Image):
            return mask.convert("L").resize(size, Image.LANCZOS)

        if not torch.is_tensor(mask):
            raise TypeError(f"Unsupported mask type: {type(mask).__name__}")

        mask = mask.detach().cpu().float()
        if mask.dim() == 4:
            mask = mask[0]
        if mask.dim() == 3:
            mask = mask[:1]
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=size[::-1], mode="bilinear", align_corners=False)
        mask = gaussian_blur(mask, kernel_size=(77, 77))
        mask = (mask[0, 0] > 0.1).float()
        return to_pil_image(mask).convert("L")

    def __call__(
        self,
        image,
        mask,
        prompt="",
        size=(512, 512),
        num_inference_steps=24,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=3.5,
        seed=42,
    ):
        """Run inpainting with the packaged diffusers pipeline."""
        if self.pipe is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        image_pil = self._image_to_pil(image, size)
        mask_pil = self._mask_to_pil(mask, size)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt or "clean realistic background",
            image=image_pil,
            mask_image=mask_pil,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        return result.resize(size, Image.LANCZOS)
