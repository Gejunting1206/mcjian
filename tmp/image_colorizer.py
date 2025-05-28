from PIL import Image

def colorize_image(input_path, output_path, target_color=(119, 171, 47)):
    """
    Colorizes non-white pixels of a black and white image to the target color.

    Args:
        input_path: Path to the input image.
        output_path: Path to save the colorized image.
        target_color: RGB tuple for the target color (default: #77AB2F).
    """
    try:
        img = Image.open(input_path).convert("RGB")
        pixels = img.load()
        width, height = img.size

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                # 检查是否接近白色，允许一些误差
                if not (r > 245 and g > 245 and b > 245):
                    pixels[x, y] = target_color

        img.save(output_path)
        print(f"图片已成功着色并保存至 {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
    except Exception as e:
        print(f"处理图片时发生错误: {e}")

if __name__ == "__main__":
    input_image_path =   # 替换为你的输入图片路径
    output_image_path = "D:/zibian/mcjian/tmp/1.20.1/assets/minecraft/textures/block/resized_images/green_oak_leaves.png"  # 替换为你的输出图片路径
    colorize_image(input_image_path, output_image_path)