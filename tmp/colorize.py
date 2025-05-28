from PIL import Image

def colorize_to_green(image_path, output_path):
    brightness_factor = 1.3  # 亮度调整系数，大于1使颜色更亮
    try:
        img = Image.open(image_path)
        # 转换为RGBA模式，确保alpha通道存在
        img = img.convert("RGBA")
        pixels = img.load()

        width, height = img.size

        for x in range(width):
            for y in range(height):
                # 获取当前像素的RGB值
                r, g, b, *a = pixels[x, y] #  如果已经是RGBA, a就是alpha通道; 如果不是, a为空列表
                gray = (r + g + b) // 3  # 或者使用更精确的灰度计算公式 gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                # 将灰度值映射到指定颜色
                r_new = int((gray / 255) * 119 * brightness_factor)
                g_new = int((gray / 255) * 171 * brightness_factor)
                b_new = int((gray / 255) * 47 * brightness_factor)
                a_new = 255 # 默认alpha值为255（不透明）
                if r_new == 0 and g_new == 0 and b_new == 0:
                    a_new = 0 # 黑色像素设置为透明
                pixels[x, y] = (r_new, g_new, b_new, a_new)

        img.save(output_path)
        print(f"图片已成功上色并保存至 {output_path}")

    except FileNotFoundError:
        print(f"找不到文件：{image_path}")
    except Exception as e:
        print(f"处理图片时发生错误：{e}")

if __name__ == "__main__":
    input_image = f"D:/zibian/mcjian/tmp/1.20.1/assets/minecraft/textures/block/resized_images/oak_leaves.png"
    output_image = f"D:/zibian/mcjian/tmp/1.20.1/assets/minecraft/textures/block/resized_images/oak_leaves_green.png"
    colorize_to_green(input_image, output_image)