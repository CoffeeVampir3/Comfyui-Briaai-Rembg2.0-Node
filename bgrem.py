import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download
import folder_paths
import comfy.model_management

# Add rembg to model paths
REMBG_MODEL_NAME = "briaai/RMBG-2.0"
MODELS_DIR = os.path.join(folder_paths.models_dir, "rembg")
os.makedirs(MODELS_DIR, exist_ok=True)

# Add the rembg directory to folder_paths
folder_paths.folder_names_and_paths["rembg"] = ([MODELS_DIR], folder_paths.supported_pt_extensions)

class BackgroundRemover:
    """Removes backgrounds from images using RMBG-2.0 model."""
    
    def __init__(self):
        self.model = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/preprocessing"

    def ensure_model_downloaded(self):
        """Ensure model files are downloaded to the correct location."""
        model_files = [
            "BiRefNet_config.py",
            "birefnet.py",
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json"
        ]
        
        for filename in model_files:
            local_path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(local_path):
                print(f"Downloading {filename} from {REMBG_MODEL_NAME}")
                hf_hub_download(
                    repo_id=REMBG_MODEL_NAME,
                    filename=filename,
                    local_dir=MODELS_DIR,
                    local_dir_use_symlinks=False
                )

    def load_model(self):
        if self.model is None:
            self.ensure_model_downloaded()
            model = AutoModelForImageSegmentation.from_pretrained(MODELS_DIR, trust_remote_code=True)
            torch.set_float32_matmul_precision('high')
            device = comfy.model_management.get_torch_device()
            model.to(device)
            model.eval()
            self.model = model
        return self.model

    def remove_background(self, image, threshold=0.5):
        device = comfy.model_management.get_torch_device()
        model = self.load_model()
        
        # Ensure we're working with batched images
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        output_images = []
        output_masks = []
        
        for i in range(image.shape[0]):
            # Convert to PIL Image - remove batch dimension and convert to uint8
            img_np = (255. * image[i].cpu().numpy()).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Setup transforms
            transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Transform and move to device
            input_tensor = transform_image(img_pil).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                preds = model(input_tensor)[-1]
                pred_mask = F.sigmoid(preds)[0, 0]
                
                # Resize mask to original size
                pred_mask = F.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0),
                    size=(img_pil.height, img_pil.width),
                    mode='bilinear',
                    align_corners=False
                )[0, 0]
                
                # Apply threshold
                mask = (pred_mask > threshold).float()
                
                # Create RGBA output - stack RGB and alpha channel
                rgb = image[i].to(device)
                alpha = mask.unsqueeze(-1).to(device)
                rgba = torch.cat([rgb, alpha], dim=-1)
                
                # Add to outputs - ensure mask has correct shape
                output_images.append(rgba.cpu())
                output_masks.append(mask.cpu())
        
        # Stack the outputs back into batches
        output_images = torch.stack(output_images)
        output_masks = torch.stack(output_masks)  # Shape: (B, H, W)
        
        return (output_images, output_masks)

NODE_CLASS_MAPPINGS = {
    "BackgroundRemover": BackgroundRemover
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundRemover": "Remove Background (RMBG-2.0)"
}