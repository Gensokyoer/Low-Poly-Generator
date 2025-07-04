import cv2

class ModeManager:
    """管理显示模式和着色风格的类"""

    def __init__(self):
        # 定义所有显示模式及其对应的键名
        self.display_modes = [
            ("low_poly", "低多边形"),
            ("feature_flow_field", "特征流场"),
            ("depth_img", "深度图"),
            ("saliency", "显著性"),
            ("dist", "距离场"),
            ("edge", "边缘检测"),
            ("original", "原图")
        ]

        # 定义所有着色风格及其对应的键名
        self.color_styles = [
            ("low_poly", "原色"),
            ("rgb_equa", "RGB均衡化"),
            ("ycrcb_equa", "YCrCb均衡化"),
            ("lab_clahe", "LAB CLAHE")
        ]

        # 初始化当前选择
        self.current_display_mode = 0
        self.current_color_style = 0

    def get_display_mode_names(self):
        """获取显示模式名称列表(用于UI显示)"""
        return [mode[1] for mode in self.display_modes]

    def get_color_style_names(self):
        """获取着色风格名称列表(用于UI显示)"""
        return [style[1] for style in self.color_styles]

    def set_display_mode(self, index):
        """设置当前显示模式"""
        if 0 <= index < len(self.display_modes):
            self.current_display_mode = index

    def set_color_style(self, index):
        """设置当前着色风格"""
        if 0 <= index < len(self.color_styles):
            self.current_color_style = index

    def is_original_mode(self):
        """判断当前是否为原图模式"""
        return self.display_modes[self.current_display_mode][0] == "original"

    def is_low_poly_mode(self):
        """判断当前是否为低多边形模式"""
        return self.display_modes[self.current_display_mode][0] == "low_poly"

    def get_image_to_display(self, process_results, original_image_path=None):
        """
        根据当前模式获取要显示的图像
        :param process_results: 处理结果字典
        :param original_image_path: 原始图像路径(仅original模式需要)
        :return: 要显示的图像或None
        """
        mode_key = self.display_modes[self.current_display_mode][0]

        # 原图模式
        if mode_key == "original":
            if original_image_path:
                return cv2.imread(original_image_path)
            return None

        # 低多边形模式(需要考虑着色风格)
        if mode_key == "low_poly":
            color_key = self.color_styles[self.current_color_style][0]
            if process_results and color_key in process_results:
                return process_results[color_key]
            return None

        # 其他处理结果模式
        if process_results and mode_key in process_results:
            return process_results[mode_key]

        return None

    def get_image_to_save(self, process_results, original_image_path=None):
        """获取要保存的图像(逻辑与显示相同)"""
        return self.get_image_to_display(process_results, original_image_path)
