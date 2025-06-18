# 输入优化器模块
# 提供鼠标平滑和输入响应优化功能

import time
from ursina import Vec2, mouse
from performance_config import PerformanceConfig

class InputOptimizer:
    """输入优化器 - 提供鼠标平滑和输入响应优化"""
    
    def __init__(self):
        # 鼠标平滑相关变量
        self.last_mouse_position = Vec2(0, 0)
        self.smoothed_mouse_position = Vec2(0, 0)
        self.last_input_time = 0
        self.input_processing_interval = PerformanceConfig.INPUT_PROCESSING_INTERVAL
        self.mouse_smoothing_factor = PerformanceConfig.MOUSE_SMOOTHING_FACTOR
        
        # 键盘响应相关变量
        self.keyboard_response_delay = PerformanceConfig.KEYBOARD_RESPONSE_DELAY
        self.key_press_times = {}
        
        # 初始化标志
        self.initialized = False
    
    def initialize(self):
        """初始化输入优化器"""
        if not self.initialized:
            self.last_mouse_position = Vec2(mouse.x, mouse.y)
            self.smoothed_mouse_position = Vec2(mouse.x, mouse.y)
            self.last_input_time = time.time()
            self.initialized = True
    
    def update(self, player, dt):
        """更新输入优化器"""
        if not self.initialized:
            self.initialize()
        
        current_time = time.time()
        
        # 鼠标平滑处理
        if current_time - self.last_input_time >= self.input_processing_interval:
            self._update_mouse_smoothing(player, current_time, dt)
    
    def _update_mouse_smoothing(self, player, current_time, dt):
        """更新鼠标平滑处理"""
        # 获取当前鼠标位置
        current_mouse = Vec2(mouse.x, mouse.y)
        
        # 使用平滑因子进行插值
        self.smoothed_mouse_position = self.smoothed_mouse_position.lerp(
            current_mouse, 
            self.mouse_smoothing_factor
        )
        
        # 计算平滑后的鼠标增量并应用到相机旋转
        if hasattr(player, 'camera_pivot') and hasattr(player, 'rotation'):
            mouse_delta = current_mouse - self.last_mouse_position
            if mouse_delta.length() > 0.001:  # 忽略微小移动，减少抖动
                # 应用平滑后的鼠标移动到玩家旋转
                player.rotation_y += mouse_delta.x * player.mouse_sensitivity[0] * dt
                player.camera_pivot.rotation_x = max(min(
                    player.camera_pivot.rotation_x - mouse_delta.y * player.mouse_sensitivity[1] * dt,
                    90
                ), -90)
        
        # 更新上次鼠标位置和输入时间
        self.last_mouse_position = current_mouse
        self.last_input_time = current_time

# 创建全局输入优化器实例
input_optimizer = InputOptimizer()