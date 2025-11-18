import torch
import numpy as np
from PIL import Image

class ImageAlphaCrop:
    """
    自动裁剪图像Alpha通道的透明边框
    可以处理四周透明和中间镂空的情况
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "alpha_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images",)
    FUNCTION = "crop_alpha"
    CATEGORY = "image/processing"
    DESCRIPTION = "根据Alpha通道自动裁剪图像的透明边框，输出带Alpha通道的图像"

    def crop_alpha(self, images, padding=0, alpha_threshold=0.01):
        """
        根据Alpha通道裁剪图像的透明边框
        
        Args:
            images: 输入图像张量 [B, H, W, C]
            padding: 裁剪后添加的额外边距
            alpha_threshold: Alpha阈值，低于此值视为透明
            
        Returns:
            cropped_images: 裁剪后的带Alpha通道的图像
        """
        
        batch_size, height, width, channels = images.shape
        cropped_images = []
        
        for i in range(batch_size):
            image_tensor = images[i]
            
            # 确保图像有Alpha通道
            if channels < 4:
                # 如果没有Alpha通道，创建一个完全不透明的Alpha
                alpha = torch.ones((height, width, 1), dtype=image_tensor.dtype)
                image_with_alpha = torch.cat([image_tensor, alpha], dim=2)
            else:
                image_with_alpha = image_tensor
            
            # 转换为numpy数组进行边界检测
            image_np = (image_with_alpha.cpu().numpy() * 255).astype(np.uint8)
            alpha_channel = image_np[:, :, 3]
            
            # 找到非透明区域的边界
            non_transparent = alpha_channel > (alpha_threshold * 255)
            
            if not np.any(non_transparent):
                # 如果整个图像都是透明的，返回原图
                cropped_images.append(image_with_alpha.unsqueeze(0))
                continue
            
            # 找到边界坐标
            rows = np.any(non_transparent, axis=1)
            cols = np.any(non_transparent, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # 添加边距
            y_min = max(0, y_min - padding)
            y_max = min(height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(width - 1, x_max + padding)
            
            # 裁剪图像
            cropped_image_np = image_np[y_min:y_max+1, x_min:x_max+1]
            
            # 转换回tensor (保持RGBA格式)
            cropped_image_tensor = torch.from_numpy(cropped_image_np.astype(np.float32) / 255.0)
            
            cropped_images.append(cropped_image_tensor.unsqueeze(0))
        
        # 合并批次
        cropped_images_batch = torch.cat(cropped_images, dim=0)
        
        return (cropped_images_batch,)


class ImageAlphaCropAdvanced:
    """
    高级Alpha通道裁剪，支持保持宽高比和指定输出尺寸
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "alpha_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "target_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 8
                }),
                "target_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 8
                }),
                "keep_aspect_ratio": (["true", "false"], {
                    "default": "true"
                }),
                "background_color": (["transparent", "black", "white"], {
                    "default": "transparent"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images",)
    FUNCTION = "crop_alpha_advanced"
    CATEGORY = "image/processing"
    DESCRIPTION = "高级Alpha通道裁剪，支持保持宽高比和调整尺寸"

    def crop_alpha_advanced(self, images, padding=0, alpha_threshold=0.01, 
                           target_width=0, target_height=0, keep_aspect_ratio="true",
                           background_color="transparent"):
        """
        高级Alpha通道裁剪功能
        """
        
        batch_size, height, width, channels = images.shape
        cropped_images = []
        
        for i in range(batch_size):
            image_tensor = images[i]
            
            # 确保图像有Alpha通道
            if channels < 4:
                alpha = torch.ones((height, width, 1), dtype=image_tensor.dtype)
                image_with_alpha = torch.cat([image_tensor, alpha], dim=2)
            else:
                image_with_alpha = image_tensor
            
            # 转换为numpy数组进行边界检测
            image_np = (image_with_alpha.cpu().numpy() * 255).astype(np.uint8)
            alpha_channel = image_np[:, :, 3]
            
            # 找到非透明区域的边界
            non_transparent = alpha_channel > (alpha_threshold * 255)
            
            if not np.any(non_transparent):
                # 如果整个图像都是透明的，返回原图
                cropped_images.append(image_with_alpha.unsqueeze(0))
                continue
            
            # 找到边界坐标
            rows = np.any(non_transparent, axis=1)
            cols = np.any(non_transparent, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # 添加边距
            y_min = max(0, y_min - padding)
            y_max = min(height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(width - 1, x_max + padding)
            
            # 裁剪图像
            cropped_image_np = image_np[y_min:y_max+1, x_min:x_max+1]
            crop_height, crop_width = cropped_image_np.shape[:2]
            
            # 调整尺寸（如果需要）
            if target_width > 0 and target_height > 0:
                if keep_aspect_ratio == "true":
                    # 保持宽高比调整尺寸
                    scale = min(target_width / crop_width, target_height / crop_height)
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    
                    # 使用PIL进行高质量缩放
                    pil_image = Image.fromarray(cropped_image_np, mode='RGBA')
                    resized_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 如果需要，创建带背景的图像
                    if background_color != "transparent" and new_width != target_width or new_height != target_height:
                        background_img = Image.new('RGBA', (target_width, target_height), 
                                                 (0, 0, 0, 0) if background_color == "transparent" else
                                                 (255, 255, 255, 255) if background_color == "white" else
                                                 (0, 0, 0, 255))
                        # 将缩放后的图像居中放置
                        x_offset = (target_width - new_width) // 2
                        y_offset = (target_height - new_height) // 2
                        background_img.paste(resized_pil, (x_offset, y_offset), resized_pil)
                        resized_pil = background_img
                    
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    cropped_image_tensor = torch.from_numpy(resized_np)
                else:
                    # 直接调整到目标尺寸
                    pil_image = Image.fromarray(cropped_image_np, mode='RGBA')
                    resized_pil = pil_image.resize((target_width, target_height), Image.LANCZOS)
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    cropped_image_tensor = torch.from_numpy(resized_np)
            else:
                # 不调整尺寸，直接转换
                cropped_image_tensor = torch.from_numpy(cropped_image_np.astype(np.float32) / 255.0)
            
            cropped_images.append(cropped_image_tensor.unsqueeze(0))
        
        # 合并批次
        cropped_images_batch = torch.cat(cropped_images, dim=0)
        
        return (cropped_images_batch,)


class ImageAddAlphaChannel:
    """
    为RGB图像添加Alpha通道
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha_value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_with_alpha",)
    FUNCTION = "add_alpha"
    CATEGORY = "image/processing"
    DESCRIPTION = "为RGB图像添加Alpha通道"

    def add_alpha(self, images, alpha_value=1.0):
        batch_size, height, width, channels = images.shape
        
        if channels >= 4:
            # 如果已经有Alpha通道，直接返回
            return (images,)
        
        # 创建Alpha通道
        alpha = torch.full((batch_size, height, width, 1), alpha_value, dtype=images.dtype)
        
        # 合并RGB和Alpha
        if channels == 3:
            images_with_alpha = torch.cat([images, alpha], dim=3)
        else:
            # 如果是单通道或双通道，先转换为RGB
            rgb_images = images.repeat(1, 1, 1, 3) if channels == 1 else images[:, :, :, :3]
            images_with_alpha = torch.cat([rgb_images, alpha], dim=3)
        
        return (images_with_alpha,)