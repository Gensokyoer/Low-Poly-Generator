import sys
import cv2
import main
import numpy as np
import time
from threading import Thread
from PySide6.QtCore import Qt, Signal, QObject, QPoint, QRect
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog,
                               QMessageBox, QGraphicsScene, QGraphicsView)
from PySide6.QtGui import QPixmap, QImage, QImageReader
from gui.ui.ui_widget import Ui_Widget
from gui.mode_manager import ModeManager
from gui.enhanced_image_viewer import EnhancedImageViewer
from stereoscopic.generate_obj import mesh_depth_to_obj

from utils.mask_convert.lasso2mask import lasso_path_to_mask
from utils.mask_convert.rec2mask import rect_item_to_mask
from utils.display_results import display_all_results


class ProgressSignals(QObject):
    update_progress = Signal(int)
    processing_done = Signal(bool, str)


class LowPolyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        # 替换默认的QGraphicsView为自己的ImageViewer
        self.viewer = EnhancedImageViewer(self)
        self.viewer.setGeometry(QRect(0, 0, 930, 800))
        self.viewer.setObjectName(u"graphicsView")

        # 存储当前图像和处理结果
        self.current_image_path = None
        self.current_img = None
        self.image_shape = None
        self.mask = None
        self.process_results = None
        self.processing_thread = None
        self.progress_signals = ProgressSignals()

        # 添加模式管理器
        self.mode_manager = ModeManager()

        # 定义默认参数
        self.default_params = {
            'blursizeSlider': 5,
            'epsilonSpinBox': 4,
            'itrateSpinBox': 10,
            'saliencyThereSpinBox': 32,
            'sampleNumSlider': 0,
            'colorizeModeBox': 0,
            'depSpinBox': 80
        }

        # 初始化UI状态
        self.init_ui_state()

        # 连接信号和槽
        self.connect_signals()

    def handle_selection_changed(self, rect):
        """处理选框变化"""
        # print(f"选框位置: {rect.x()}, {rect.y()}, 大小: {rect.width()}x{rect.height()}")
        pass

    def handle_mask_updated(self, mask_image):
        """处理掩码更新"""
        # 可以将掩码转换为numpy数组进行进一步处理
        pass

    def init_ui_state(self):
        """初始化UI控件的默认状态"""
        # 设置所有参数为默认值
        self.reset_parameters(silent=True)

        # 设置下拉框默认选项
        self.ui.outputModeBox.clear()
        self.ui.outputModeBox.addItems(self.mode_manager.get_display_mode_names())

        self.ui.colorizeModeBox.clear()
        self.ui.colorizeModeBox.addItems(self.mode_manager.get_color_style_names())

        # 设置标签文本居中
        self.ui.blurValueLabel.setAlignment(Qt.AlignCenter)
        self.ui.sampleValueLabel.setAlignment(Qt.AlignCenter)

        # 初始化滑块值显示
        self.update_slider_value_display()

        # 初始化选框模式
        self.ui.dragToolButton.setChecked(True)
        self.viewer.set_tool("pan")




    def connect_signals(self):
        """连接信号和槽函数"""
        self.ui.loadButton.clicked.connect(self.load_image)
        self.ui.saveButton.clicked.connect(self.save_image)
        self.ui.applyButton.clicked.connect(self.apply_processing)
        self.ui.resetButton.clicked.connect(self.reset_parameters)
        self.ui.saveModelButton.clicked.connect(self.save_model)
        # self.ui.saveSelectionButton.clicked.connect(self.save_selection)
        # self.ui.selectAllButton.clicked.connect(self.select_all_image)

        # 选框模式
        self.ui.lassoToolButton.clicked.connect(lambda: self.viewer.set_tool("lasso"))
        self.ui.recToolButton.clicked.connect(lambda: self.viewer.set_tool("rectangle"))
        self.ui.dragToolButton.clicked.connect(lambda: self.viewer.set_tool("pan"))

        # 输出模式改变时更新显示
        self.ui.outputModeBox.currentIndexChanged.connect(
            lambda i: self.mode_manager.set_display_mode(i) or self.update_display()
        )
        # 着色风格改变时更新显示
        self.ui.colorizeModeBox.currentIndexChanged.connect(
            lambda i: self.mode_manager.set_color_style(i) or self.update_colorization()
        )

        # 连接进度信号
        self.progress_signals.update_progress.connect(self.ui.progressBar.setValue)
        self.progress_signals.processing_done.connect(self.on_processing_done)

        # 连接滑块值改变信号
        self.ui.blursizeSlider.valueChanged.connect(self.update_blursize_value)
        self.ui.sampleNumSlider.valueChanged.connect(self.update_sample_num_value)

        # 连接工具完成信号
        lasso_tool = self.viewer._tool_manager._tools["lasso"]
        rect_tool = self.viewer._tool_manager._tools["rectangle"]

        lasso_tool.finished.connect(self.handle_lasso_finished)
        rect_tool.finished.connect(self.handle_rect_finished)
        lasso_tool.be_reset.connect(self.clear_selection)
        rect_tool.be_reset.connect(self.clear_selection)

    def handle_lasso_finished(self, item):
        self.mask = lasso_path_to_mask(item, self.image_shape)
        print("套索选区已更新")

    def handle_rect_finished(self, item):
        self.mask = rect_item_to_mask(item, self.image_shape)
        print("矩形选区已更新")

    def clear_selection(self):
        """清除当前选区和mask"""
        if self.image_shape:
            self.mask = np.zeros(self.image_shape, dtype=np.uint8)

    def update_slider_value_display(self):
        """更新滑块值显示"""
        self.ui.blurValueLabel.setText(str(self.ui.blursizeSlider.value()))
        self.ui.sampleValueLabel.setText(str(self.ui.sampleNumSlider.value()))

    def update_blursize_value(self, value):
        """更新模糊大小滑块值显示"""
        self.ui.blurValueLabel.setText(str(value))

    def update_sample_num_value(self, value):
        """更新采样数量滑块值显示"""
        self.ui.sampleValueLabel.setText(str(value))

    def load_image(self):
        """导入图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像文件",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if file_path:
            try:
                # 检查图像是否可读
                reader = QImageReader(file_path)
                if not reader.canRead():
                    raise ValueError("无法读取图像文件")

                # 存储当前图像路径
                self.current_image_path = file_path

                # 加载图像用于显示
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    raise ValueError("无法加载图像文件")

                self.current_img = cv2.imread(self.current_image_path)
                self.image_shape = self.current_img.shape[:2]
                self.mask = np.zeros(self.image_shape, dtype=np.uint8)

                # 在graphicsView中显示图像
                self.display_image(pixmap)

                # 重置处理结果
                self.process_results = None

                # 更新状态
                self.ui.progressBar.setValue(0)

                # 启用按钮
                self.ui.recToolButton.setEnabled(True)
                self.ui.lassoToolButton.setEnabled(True)
                self.ui.applyButton.setEnabled(True)
                QMessageBox.information(self, "成功", "图像加载成功")

            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载图像失败: {str(e)}")
                # self.current_image_path = None

    def save_image(self):
        """导出图像"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return

            # 获取要保存的图像
        image_to_save = self.mode_manager.get_image_to_save(
            self.process_results,
            self.current_image_path
        )

        if image_to_save is None:
            QMessageBox.warning(self, "警告", "没有可保存的处理结果")
            return

        # 打开文件对话框选择保存位置
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图像",
            "",
            "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;位图 (*.bmp)"
        )

        if file_path:
            try:
                if not cv2.imwrite(file_path, image_to_save):
                    raise ValueError("保存失败")
                QMessageBox.information(self, "成功", "图像保存成功")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存图像失败: {str(e)}")

    def save_model(self):
        """导出模型文件"""
        if self.process_results is None:
            QMessageBox.warning(self, "警告", "没有可保存的模型，请先处理图像")
            return

        # 从处理结果中获取所需数据
        try:
            mesh = self.process_results['mesh']
            depth = self.process_results['depth']
            low_poly = self.process_results['low_poly']
        except KeyError:
            QMessageBox.warning(self, "错误", "模型数据不完整，请重新处理图像")
            return

        # 获取深度最大值参数
        depth_max = self.ui.depSpinBox.value()

        # 弹出保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存模型文件",
            "",
            "OBJ模型文件 (*.obj)"
        )

        if not file_path:  # 用户取消保存
            return

        try:
            # 调用模型导出函数
            mesh_depth_to_obj(
                mesh,
                depth,
                low_poly,
                depth_max=depth_max,
                output_path=file_path
            )
            QMessageBox.information(self, "成功", "模型保存成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存模型失败: {str(e)}")


    def apply_processing(self):
        """应用图像处理"""
        if not self.current_image_path or self.mask is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        if self.processing_thread and self.processing_thread.is_alive():
            QMessageBox.warning(self, "警告", "正在处理中，请稍候")
            return

        # 检查mask是否全0
        if isinstance(self.mask, np.ndarray) and not np.any(self.mask):
            self.mask = np.ones(self.image_shape, dtype=np.uint8) * 255

        # 获取参数值
        blur_size = self.ui.blursizeSlider.value()
        epsilon = self.ui.epsilonSpinBox.value()
        relaxation_iterations = self.ui.itrateSpinBox.value()
        saliency_thres = self.ui.saliencyThereSpinBox.value()
        deltaN = self.ui.sampleNumSlider.value()


        # 禁用按钮防止重复点击
        self.ui.applyButton.setEnabled(False)
        self.ui.loadButton.setEnabled(False)
        self.ui.saveButton.setEnabled(False)
        self.ui.saveModelButton.setEnabled(False)
        self.ui.recToolButton.setEnabled(False)
        self.ui.lassoToolButton.setEnabled(False)
        self.ui.dragToolButton.setEnabled(False)

        # 启动进度条更新线程
        self.progress_signals.update_progress.emit(0)
        self.processing_thread = Thread(
            target=self.process_image,
            args=(blur_size, epsilon, relaxation_iterations,
                  saliency_thres, deltaN)
        )
        self.processing_thread.start()

    def select_all_image(self):
        """处理全选按钮点击事件"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        # 调用viewer的全选方法
        if not self.viewer.select_all():
            QMessageBox.warning(self, "警告", "无法全选，可能没有加载图像")

    def save_selection(self):
        pass

    def process_image(self, blur_size, epsilon, relaxation_iterations, saliency_thres, deltaN):
        """在新线程中处理图像"""
        start_time = time.time()
        estimated_duration = 8  # 预计8秒完成

        # 启动进度条更新
        progress_thread = Thread(target=self.update_progress, args=(start_time, estimated_duration))
        progress_thread.start()

        try:
            # 调用主处理函数
            self.process_results = main.get_selected_region_imgs(
                img = self.current_img,
                mask = self.mask,
                blur_size=blur_size,
                epsilon=epsilon,
                relaxation_iterations=relaxation_iterations,
                saliency_thres=saliency_thres,
                deltaN=deltaN
            )

            # 处理完成，通知主线程
            self.progress_signals.processing_done.emit(True, "图像处理完成")

        except Exception as e:
            self.progress_signals.processing_done.emit(False, f"图像处理失败: {str(e)}")
        finally:
            progress_thread.join()

    def update_progress(self, start_time, estimated_duration):
        """更新进度条"""
        while True:
            elapsed = time.time() - start_time
            progress = min(99, int((elapsed / estimated_duration) * 100))
            self.progress_signals.update_progress.emit(progress)

            # 如果处理完成或超时，退出循环
            if progress >= 99 or (self.processing_thread and not self.processing_thread.is_alive()):
                break

            time.sleep(0.1)  # 更新间隔

    def on_processing_done(self, success, message):
        """处理完成时的回调"""
        # 更新进度条到100%
        self.progress_signals.update_progress.emit(100)

        # 启用按钮
        self.ui.loadButton.setEnabled(True)
        self.ui.saveButton.setEnabled(True)
        self.ui.saveModelButton.setEnabled(True)
        self.ui.dragToolButton.setEnabled(True)

        # 恢复拖拽模式
        self.ui.dragToolButton.setChecked(True)
        self.viewer.set_tool("pan")

        # 更新显示
        if success:
            self.update_display()
            self.update_colorization()
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "错误", message)

    def update_display(self):
        """根据当前输出模式更新显示"""
        if not self.current_image_path:
            return

        # 获取当前模式对应的图像
        image = self.mode_manager.get_image_to_display(
            self.process_results,
            self.current_image_path
        )

        if image is not None:
            self.display_cv_image(image)

    def update_colorization(self):
        """更新着色风格显示"""
        if self.mode_manager.is_low_poly_mode():
            self.update_display()

    def display_cv_image(self, cv_img):
        """显示OpenCV图像"""
        if len(cv_img.shape) == 2:  # 灰度图
            qimage = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0],
                            cv_img.strides[0], QImage.Format_Grayscale8)
        else:  # 彩色图
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        self.display_image(pixmap)

    def display_image(self, pixmap):
        """在graphicsView中显示图像"""
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.viewer.setScene(scene)
        self.viewer.fit_view()

    def reset_parameters(self, silent=False):
        """重置参数为默认值"""
        # 设置滑块和数值输入框的默认值
        for widget_name, value in self.default_params.items():
            widget = getattr(self.ui, widget_name)
            if hasattr(widget, 'setValue'):
                widget.setValue(value)
            elif hasattr(widget, 'setCurrentIndex'):
                widget.setCurrentIndex(value)

        # 更新滑块值显示
        self.update_slider_value_display()

        if not silent:
            QMessageBox.information(self, "提示", "参数已重置为默认值")

    def resizeEvent(self, event):
        """窗口大小改变时自动调整视图"""
        super().resizeEvent(event)
        if self.viewer.scene():
            self.viewer.fit_view()