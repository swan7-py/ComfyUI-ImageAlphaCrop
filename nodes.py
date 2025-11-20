import torch
import numpy as np
import os
import json
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import folder_paths
import node_helpers

class LoadImagesFromPathRGBA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE PATH")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "Swan"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        dir_files = dir_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
                
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                
                # 转换为RGBA模式
                if i.mode != 'RGBA':
                    image = i.convert("RGBA")
                else:
                    image = i
                
                # 转换为numpy数组
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]

                # 提取Alpha通道作为mask
                mask_np = image_np[:, :, 3]
                mask_tensor = torch.from_numpy(mask_np)

                images.append(image_tensor)
                masks.append(mask_tensor)
                file_paths.append(str(image_path))
                image_count += 1
                
            except Exception as e:
                print(f"Failed to load image {image_path}: {str(e)}")
                continue

        if len(images) == 0:
            raise Exception(f"No valid images found in directory '{directory}' after filtering and starting from index {start_index}")

        return (images, masks, file_paths)

class SaveImageRGBA:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The RGBA images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_RGBA", "tooltip": "The prefix for the file to save."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_rgba"

    OUTPUT_NODE = True

    CATEGORY = "Swan"
    DESCRIPTION = "Saves the input RGBA images as PNG with transparency to your ComfyUI output directory."

    def save_images_rgba(self, images, filename_prefix="ComfyUI_RGBA", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        
        for (batch_number, image) in enumerate(images):
            # 确保图像有4个通道 (RGBA)
            if image.shape[2] < 4:
                # 如果图像没有Alpha通道，添加一个不透明的Alpha通道
                height, width = image.shape[0], image.shape[1]
                alpha = torch.ones((height, width, 1), dtype=image.dtype)
                image_rgba = torch.cat([image, alpha], dim=2)
            else:
                image_rgba = image
            
            # 转换为numpy数组并缩放到0-255范围
            i = 255. * image_rgba.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8), 'RGBA')
            
            metadata = None
            if not hasattr(self, 'disable_metadata') or not self.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class LoadImageWithAlpha:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Swan"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "load_image"
    
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            # 始终转换为RGBA模式
            if i.mode != 'RGBA':
                image = i.convert("RGBA")
            else:
                image = i

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            # 将RGBA图像转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            # 提取Alpha通道作为mask
            mask_np = image_np[:, :, 3]
            mask_tensor = torch.from_numpy(mask_np)
            
            output_images.append(image_tensor)
            output_masks.append(mask_tensor.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, w, h)


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
    CATEGORY = "Swan"
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
                "transparent_images": ("IMAGE",),  # 修改显示名称为"透明图像"
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
            "optional": {
                "original_images": ("IMAGE",),  
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images",)
    FUNCTION = "crop_alpha_advanced"
    CATEGORY = "Swan"
    DESCRIPTION = "高级Alpha通道裁剪，支持保持宽高比和调整尺寸"

    def crop_alpha_advanced(self, transparent_images, padding=0, alpha_threshold=0.01, 
                           target_width=0, target_height=0, keep_aspect_ratio="true",
                           background_color="transparent", original_images=None):
        """
        高级Alpha通道裁剪功能
        
        Args:
            transparent_images: 透明图像，用于确定裁剪区域
            original_images: 原始图像，可选，如果提供则裁剪此图像
            其他参数保持不变
        """
        
        batch_size, height, width, channels = transparent_images.shape
        cropped_images = []
        
        # 如果没有提供原始图像，则使用透明图像
        if original_images is None:
            source_images = transparent_images
        else:
            source_images = original_images
            
        source_batch_size = source_images.shape[0]
        
        for i in range(batch_size):
            source_idx = i if i < source_batch_size else source_batch_size - 1
            
            transparent_image = transparent_images[i]
            source_image = source_images[source_idx]
            
            # 确保源图像有Alpha通道（如果是RGB，添加不透明Alpha通道）
            source_channels = source_image.shape[2]
            if source_channels < 4:
                source_height, source_width = source_image.shape[0], source_image.shape[1]
                alpha = torch.ones((source_height, source_width, 1), dtype=source_image.dtype)
                source_with_alpha = torch.cat([source_image, alpha], dim=2)
            else:
                source_with_alpha = source_image
            
            # 确保透明图像有Alpha通道
            transparent_channels = transparent_image.shape[2]
            if transparent_channels < 4:
                transparent_height, transparent_width = transparent_image.shape[0], transparent_image.shape[1]
                alpha = torch.ones((transparent_height, transparent_width, 1), dtype=transparent_image.dtype)
                transparent_with_alpha = torch.cat([transparent_image, alpha], dim=2)
            else:
                transparent_with_alpha = transparent_image
            
            # 调整源图像尺寸以匹配透明图像
            if source_with_alpha.shape[0] != transparent_with_alpha.shape[0] or source_with_alpha.shape[1] != transparent_with_alpha.shape[1]:
                # 使用PIL进行高质量缩放
                source_np = (source_with_alpha.cpu().numpy() * 255).astype(np.uint8)
                transparent_np = (transparent_with_alpha.cpu().numpy() * 255).astype(np.uint8)
                
                source_pil = Image.fromarray(source_np, mode='RGBA')
                target_size = (transparent_np.shape[1], transparent_np.shape[0])
                resized_source_pil = source_pil.resize(target_size, Image.LANCZOS)
                resized_source_np = np.array(resized_source_pil).astype(np.float32) / 255.0
                source_with_alpha = torch.from_numpy(resized_source_np)
            
            # 使用透明图像的Alpha通道确定裁剪区域
            transparent_np = (transparent_with_alpha.cpu().numpy() * 255).astype(np.uint8)
            alpha_channel = transparent_np[:, :, 3]
            
            # 找到非透明区域的边界
            non_transparent = alpha_channel > (alpha_threshold * 255)
            
            if not np.any(non_transparent):
                # 如果整个图像都是透明的，返回原图
                cropped_images.append(source_with_alpha.unsqueeze(0))
                continue
            
            # 找到边界坐标
            rows = np.any(non_transparent, axis=1)
            cols = np.any(non_transparent, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # 添加边距
            y_min = max(0, y_min - padding)
            y_max = min(transparent_np.shape[0] - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(transparent_np.shape[1] - 1, x_max + padding)
            
            # 裁剪源图像
            source_np = (source_with_alpha.cpu().numpy() * 255).astype(np.uint8)
            cropped_image_np = source_np[y_min:y_max+1, x_min:x_max+1]
            crop_height, crop_width = cropped_image_np.shape[:2]
            
            # 调整尺寸（如果需要）
            if target_width > 0 and target_height > 0:
                if keep_aspect_ratio == "true":
                    # 保持宽高比调整尺寸
                    scale = min(target_width / crop_width, target_height / crop_height)
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    
                    pil_image = Image.fromarray(cropped_image_np, mode='RGBA')
                    resized_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 如果需要，创建带背景的图像
                    if background_color != "transparent" and new_width != target_width or new_height != target_height:
                        background_img = Image.new('RGBA', (target_width, target_height), 
                                                 (0, 0, 0, 0) if background_color == "transparent" else
                                                 (255, 255, 255, 255) if background_color == "white" else
                                                 (0, 0, 0, 255))
                        x_offset = (target_width - new_width) // 2
                        y_offset = (target_height - new_height) // 2
                        background_img.paste(resized_pil, (x_offset, y_offset), resized_pil)
                        resized_pil = background_img
                    
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    cropped_image_tensor = torch.from_numpy(resized_np)
                else:
                    pil_image = Image.fromarray(cropped_image_np, mode='RGBA')
                    resized_pil = pil_image.resize((target_width, target_height), Image.LANCZOS)
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    cropped_image_tensor = torch.from_numpy(resized_np)
            else:
                cropped_image_tensor = torch.from_numpy(cropped_image_np.astype(np.float32) / 255.0)
            
            cropped_images.append(cropped_image_tensor.unsqueeze(0))
        
        cropped_images_batch = torch.cat(cropped_images, dim=0)
        
        return (cropped_images_batch,)
    

NODE_CLASS_MAPPINGS = {
    "LoadImageWithAlpha": LoadImageWithAlpha,
    "ImageAlphaCrop": ImageAlphaCrop,
    "ImageAlphaCropAdvanced": ImageAlphaCropAdvanced,
    "SaveImageRGBA": SaveImageRGBA,
    "LoadImagesFromPathRGBA": LoadImagesFromPathRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithAlpha": "Load Image (With Alpha)",
    "ImageAlphaCrop": "Image Alpha Crop",
    "ImageAlphaCropAdvanced": "Image Alpha Crop (Advanced)",
    "SaveImageRGBA": "Save Image (RGBA)",
    "LoadImagesFromPathRGBA": "Load Images From Path (RGBA)",
}
