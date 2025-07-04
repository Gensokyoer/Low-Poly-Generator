import numpy as np
import cv2
import os

def display_all_results(result):
    """
    显示result字典中的所有图像

    参数:
        result: 包含各种处理结果的字典，格式与get_all_imgs或get_selected_region_imgs返回的一致
    """
    # 获取所有可用的图像键
    available_keys = [key for key in result.keys() if isinstance(result[key], np.ndarray)]

    # 设置窗口布局参数
    max_width = 800  # 窗口最大宽度
    max_height = 600  # 窗口最大高度

    # 创建显示函数
    def show_image(title, img):
        # 调整图像大小以适应显示
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)  # 保持比例，不超过最大尺寸

        # 彩色图像需要转换为BGR格式显示
        if len(img.shape) == 3 and img.shape[2] == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.dtype == np.float32 else img.copy()
        else:
            # 单通道图像转换为伪彩色以便更好地可视化
            display_img = cv2.applyColorMap((img * 255).astype(np.uint8),
                                            cv2.COLORMAP_JET) if img.dtype == np.float32 else img

        resized_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))
        cv2.imshow(title, resized_img)

    # 显示所有可用图像
    for key in available_keys:
        show_image(key, result[key])

        img = result[key]
        # 处理图像格式转换
        if len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像
            if img.dtype == np.float32:
                save_img = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
            else:
                save_img = img.copy()
        else:  # 单通道图像
            if img.dtype == np.float32:
                save_img = (img * 255).astype(np.uint8)
            else:
                save_img = img

        # 获取桌面路径（跨平台兼容）
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

        # 创建保存目录（如果不存在）
        save_dir = os.path.join(desktop_path, 'Processing_Results')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{key.replace(' ', '_')}.png"  # 替换空格，使用PNG格式保持质量
        filepath = os.path.join(save_dir, filename)

        # 保存图像
        cv2.imwrite(filepath, save_img)

    # 等待按键关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()