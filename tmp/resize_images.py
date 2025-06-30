from PIL import Image
import os

def resize_images(image_dir):
    # 创建一个新文件夹来保存调整大小后的图片
    output_dir = os.path.join(image_dir, "resized_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹中的所有文件
    for filename in ['sun.png','moon.png']:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 添加更多图片格式
            try:
                # 打开图片
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path)

                # 调整图片大小
                img = img.resize((128, 128), Image.NEAREST)

                # 保存调整大小后的图片到新文件夹
                output_path = os.path.join(output_dir, filename)
                img.save(output_path)
                print(f"已调整大小并保存: {filename}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    image_dir = f"..\\src\\assets"
    resize_images(image_dir)