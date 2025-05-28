from PIL import Image, ImageFilter
import os

def blur_background(image_path, output_path, blur_radius=10):
    """
    对背景图片应用高斯模糊效果
    
    Args:
        image_path: 原始图片路径
        output_path: 模糊后图片保存路径
        blur_radius: 模糊半径，值越大越模糊
    """
    try:
        # 打开原始图片
        img = Image.open(image_path)
        
        # 应用高斯模糊
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 保存模糊后的图片
        blurred_img.save(output_path)
        print(f"模糊背景已保存至 {output_path}")
        
    except FileNotFoundError:
        print(f"找不到文件：{image_path}")
    except Exception as e:
        print(f"处理图片时发生错误：{e}")

if __name__ == "__main__":
    # 原始背景图片路径
    original_bg = "d:\\zibian\\mcjian\\assets\\background.png"
    
    # 模糊后图片保存路径
    blurred_bg = "d:\\zibian\\mcjian\\assets\\background_blurred.png"
    
    # 应用模糊效果
    blur_background(original_bg, blurred_bg, blur_radius=15)