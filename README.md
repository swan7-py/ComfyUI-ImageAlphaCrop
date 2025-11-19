# ComfyUI Image Alpha Crop Plugin

一个用于自动裁剪图像Alpha通道透明边框的ComfyUI插件。特别适合处理抠图后的Logo、图标等需要去除透明边框的图像。

## 功能特点

- 🔍 **智能透明检测**：自动识别并裁剪四周透明区域
- ✂️ **精确裁剪**：支持图像中间有镂空的情况
- 🎨 **保持Alpha通道**：输出带透明背景的RGBA图像
- 📐 **尺寸调整**：支持调整输出尺寸和保持宽高比
- 🎯 **批量处理**：一次处理多张图像
- ⚙️ **灵活参数**：可调节边距和透明度阈值

## 安装方法

1.  ComfyUI 的 `custom_nodes` 目录下
2.   `git clone https://github.com/swan7-py/ComfyUI-ImageAlphaCrop`
3. 重启 ComfyUI

## 节点介绍

### 1. Image Alpha Crop (基本裁剪)

**输入参数：**
- `images`：输入图像（支持带Alpha通道的图像）
- `padding`：裁剪后添加的边距像素数（默认：0）
- `alpha_threshold`：Alpha阈值，低于此值视为透明（默认：0.01）

**输出：**
- `cropped_images`：裁剪后的带Alpha通道图像（RGBA格式）

### 2. Image Alpha Crop (Advanced) (高级裁剪)

**输入参数：**
- `images`：输入图像
- `padding`：裁剪边距（默认：0）
- `alpha_threshold`：透明度阈值（默认：0.01）
- `target_width`：目标宽度（0表示不调整）
- `target_height`：目标高度（0表示不调整）
- `keep_aspect_ratio`：是否保持宽高比（默认：true）
- `background_color`：背景颜色（透明/黑色/白色）

**输出：**
- `cropped_images`：裁剪并调整后的带Alpha通道图像

## 适用场景

- 🏢 **Logo处理**：去除抠图后四周的透明区域
- 🎨 **图标优化**：自动裁剪图标到最小尺寸
- 🖼️ **UI元素**：准备用于UI设计的透明图像
- 📱 **应用开发**：为移动应用准备图像资源
- 🌐 **网页设计**：优化网页用透明图像
