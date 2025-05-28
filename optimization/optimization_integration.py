# 优化集成模块 - 整合所有优化技术
from ursina import *
import time
from math import floor

# 导入优化模块
from .chunk_optimization import ChunkManager, DistanceCheckOptimizer
from .instanced_rendering import InstancedRenderer, MeshCombiner

class OptimizationManager:
    """优化管理器 - 整合所有优化技术并提供简单的接口"""
    
    def __init__(self, chunk_size=8, render_distance=2):
        # 初始化各种优化器
        self.chunk_manager = ChunkManager(chunk_size=chunk_size, render_distance=render_distance)
        self.instanced_renderer = InstancedRenderer()
        self.mesh_combiner = MeshCombiner(chunk_size=chunk_size)
        self.distance_optimizer = DistanceCheckOptimizer(check_interval=30)
        
        # 优化配置
        self.use_instanced_rendering = True  # 是否使用实例化渲染
        self.use_mesh_combining = True      # 是否使用网格合并
        self.use_distance_culling = True     # 是否使用距离剔除
        self.use_chunk_management = True     # 是否使用区块管理
        
        # 性能监控
        self.last_update_time = time.time()
        self.update_interval = 0.5  # 更新间隔（秒）
        self.fps_history = []
        self.current_fps = 0
        
        # 自适应优化参数
        self.adaptive_optimization = True  # 是否启用自适应优化
        self.target_fps = 30              # 目标帧率
        
    def add_block(self, block):
        """添加方块到所有启用的优化系统"""
        if self.use_chunk_management:
            self.chunk_manager.add_block_to_chunk(block)
        
        if self.use_instanced_rendering:
            self.instanced_renderer.add_block(block)
        
        if self.use_mesh_combining:
            self.mesh_combiner.add_block(block)
    
    def remove_block(self, block):
        """从所有启用的优化系统中移除方块"""
        if self.use_chunk_management:
            self.chunk_manager.remove_block_from_chunk(block)
        
        if self.use_instanced_rendering:
            self.instanced_renderer.remove_block(block)
        
        if self.use_mesh_combining:
            self.mesh_combiner.remove_block(block)
    
    def update(self, player, all_blocks):
        """更新所有优化系统"""
        current_time = time.time()
        
        # 更新当前帧率
        if hasattr(application, 'fps'):
            self.current_fps = application.fps
            self.fps_history.append(self.current_fps)
            # 只保留最近10个帧率记录
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
        
        # 自适应优化 - 根据帧率调整优化参数
        if self.adaptive_optimization and self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self._adjust_optimization_params(avg_fps)
        
        # 降低更新频率
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 更新区块管理
        if self.use_chunk_management and player:
            self.chunk_manager.update_chunks_visibility(player.position)
        
        # 更新距离剔除
        if self.use_distance_culling and player:
            self.distance_optimizer.update(player, all_blocks)
        
        # 更新实例化渲染
        if self.use_instanced_rendering:
            self.instanced_renderer.update()
        
        # 更新网格合并
        if self.use_mesh_combining:
            # 构建按区块分组的方块字典
            blocks_by_chunk = {}
            for block in all_blocks:
                chunk_coords = self.mesh_combiner._get_chunk_coords(block.position)
                if chunk_coords not in blocks_by_chunk:
                    blocks_by_chunk[chunk_coords] = []
                blocks_by_chunk[chunk_coords].append(block)
            
            self.mesh_combiner.update(blocks_by_chunk)
    
    def _adjust_optimization_params(self, avg_fps):
        """根据帧率自适应调整优化参数"""
        # 如果帧率太低，启用更多优化
        if avg_fps < self.target_fps * 0.7:  # 低于目标帧率的70%
            # 降低渲染距离
            if self.chunk_manager.render_distance > 1:
                self.chunk_manager.render_distance -= 1
            
            # 启用实例化渲染
            self.use_instanced_rendering = True
            
            # 增加距离检查间隔
            self.distance_optimizer.check_interval = min(60, self.distance_optimizer.check_interval + 10)
        
        # 如果帧率很高，可以减少优化以提高视觉质量
        elif avg_fps > self.target_fps * 1.5:  # 高于目标帧率的150%
            # 增加渲染距离
            self.chunk_manager.render_distance += 1
            
            # 减少距离检查间隔
            self.distance_optimizer.check_interval = max(10, self.distance_optimizer.check_interval - 5)
    
    def toggle_instanced_rendering(self):
        """切换实例化渲染"""
        self.use_instanced_rendering = not self.use_instanced_rendering
        if not self.use_instanced_rendering:
            # 恢复所有方块的原始视觉效果
            self.instanced_renderer.restore_blocks()
        return self.use_instanced_rendering
    
    def toggle_mesh_combining(self):
        """切换网格合并"""
        self.use_mesh_combining = not self.use_mesh_combining
        return self.use_mesh_combining
    
    def toggle_distance_culling(self):
        """切换距离剔除"""
        self.use_distance_culling = not self.use_distance_culling
        return self.use_distance_culling
    
    def toggle_chunk_management(self):
        """切换区块管理"""
        self.use_chunk_management = not self.use_chunk_management
        return self.use_chunk_management
    
    def toggle_adaptive_optimization(self):
        """切换自适应优化"""
        self.adaptive_optimization = not self.adaptive_optimization
        return self.adaptive_optimization

# 优化设置UI界面
class OptimizationSettingsUI:
    """优化设置UI界面 - 提供图形界面控制所有优化选项"""
    
    def __init__(self, optimization_manager, parent=None):
        self.optimization_manager = optimization_manager
        
        # 创建UI容器
        self.ui_parent = parent or scene
        self.window = WindowPanel(
            title='优化设置',
            color=color.dark_gray.tint(-.2),
            scale=(0.4, 0.6),
            position=(0.6, 0),
            parent=self.ui_parent
        )
        
        # 默认隐藏窗口
        self.window.visible = False
        
        # 创建设置按钮
        self.settings_button = Button(
            text='优化',
            color=color.azure,
            scale=(0.1, 0.05),
            position=(0.85, 0.45),
            parent=camera.ui
        )
        self.settings_button.tooltip = Tooltip('打开优化设置')
        self.settings_button.on_click = self.toggle_window
        
        # 创建UI元素
        self._create_ui_elements()
        
        # 更新UI状态
        self.update_ui_state()
    
    def _create_ui_elements(self):
        """创建所有UI元素"""
        y_offset = 0.22
        spacing = 0.06
        
        # 实例化渲染开关
        self.instanced_rendering_toggle = ButtonGroup(['启用', '禁用'], default='启用' if self.optimization_manager.use_instanced_rendering else '禁用')
        self.instanced_rendering_toggle.y = y_offset
        self.instanced_rendering_toggle.parent = self.window
        self.instanced_rendering_toggle.on_value_changed = self._on_instanced_rendering_changed
        Text('实例化渲染:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        
        # 网格合并开关
        y_offset -= spacing
        self.mesh_combining_toggle = ButtonGroup(['启用', '禁用'], default='启用' if self.optimization_manager.use_mesh_combining else '禁用')
        self.mesh_combining_toggle.y = y_offset
        self.mesh_combining_toggle.parent = self.window
        self.mesh_combining_toggle.on_value_changed = self._on_mesh_combining_changed
        Text('网格合并:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        
        # 距离剔除开关
        y_offset -= spacing
        self.distance_culling_toggle = ButtonGroup(['启用', '禁用'], default='启用' if self.optimization_manager.use_distance_culling else '禁用')
        self.distance_culling_toggle.y = y_offset
        self.distance_culling_toggle.parent = self.window
        self.distance_culling_toggle.on_value_changed = self._on_distance_culling_changed
        Text('距离剔除:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        
        # 区块管理开关
        y_offset -= spacing
        self.chunk_management_toggle = ButtonGroup(['启用', '禁用'], default='启用' if self.optimization_manager.use_chunk_management else '禁用')
        self.chunk_management_toggle.y = y_offset
        self.chunk_management_toggle.parent = self.window
        self.chunk_management_toggle.on_value_changed = self._on_chunk_management_changed
        Text('区块管理:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        
        # 自适应优化开关
        y_offset -= spacing
        self.adaptive_optimization_toggle = ButtonGroup(['启用', '禁用'], default='启用' if self.optimization_manager.adaptive_optimization else '禁用')
        self.adaptive_optimization_toggle.y = y_offset
        self.adaptive_optimization_toggle.parent = self.window
        self.adaptive_optimization_toggle.on_value_changed = self._on_adaptive_optimization_changed
        Text('自适应优化:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        
        # 渲染距离滑块
        y_offset -= spacing
        Text('渲染距离:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        self.render_distance_slider = Slider(min=1, max=5, step=1, default=self.optimization_manager.chunk_manager.render_distance,
                                           dynamic=True, parent=self.window, y=y_offset)
        self.render_distance_slider.on_value_changed = self._on_render_distance_changed
        self.render_distance_text = Text(text=str(self.render_distance_slider.value), parent=self.window, 
                                       y=y_offset, x=0.2, scale=0.7)
        
        # 目标帧率滑块
        y_offset -= spacing
        Text('目标帧率:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        self.target_fps_slider = Slider(min=15, max=60, step=5, default=self.optimization_manager.target_fps,
                                      dynamic=True, parent=self.window, y=y_offset)
        self.target_fps_slider.on_value_changed = self._on_target_fps_changed
        self.target_fps_text = Text(text=str(self.target_fps_slider.value), parent=self.window, 
                                  y=y_offset, x=0.2, scale=0.7)
        
        # 当前帧率显示
        y_offset -= spacing
        Text('当前帧率:', parent=self.window, y=y_offset, x=-0.15, scale=0.7)
        self.current_fps_text = Text(text='0', parent=self.window, y=y_offset, x=0.1, scale=0.7)
        
        # 关闭按钮
        y_offset -= spacing * 1.5
        self.close_button = Button(text='关闭', parent=self.window, y=y_offset, scale=(0.3, 0.05))
        self.close_button.on_click = self.hide_window
    
    def _on_instanced_rendering_changed(self):
        """实例化渲染开关回调"""
        enabled = self.instanced_rendering_toggle.value == '启用'
        self.optimization_manager.use_instanced_rendering = enabled
        if not enabled:
            self.optimization_manager.instanced_renderer.restore_blocks()
    
    def _on_mesh_combining_changed(self):
        """网格合并开关回调"""
        self.optimization_manager.use_mesh_combining = self.mesh_combining_toggle.value == '启用'
    
    def _on_distance_culling_changed(self):
        """距离剔除开关回调"""
        self.optimization_manager.use_distance_culling = self.distance_culling_toggle.value == '启用'
    
    def _on_chunk_management_changed(self):
        """区块管理开关回调"""
        self.optimization_manager.use_chunk_management = self.chunk_management_toggle.value == '启用'
    
    def _on_adaptive_optimization_changed(self):
        """自适应优化开关回调"""
        self.optimization_manager.adaptive_optimization = self.adaptive_optimization_toggle.value == '启用'
    
    def _on_render_distance_changed(self):
        """渲染距离滑块回调"""
        value = int(self.render_distance_slider.value)
        self.optimization_manager.chunk_manager.render_distance = value
        self.render_distance_text.text = str(value)
    
    def _on_target_fps_changed(self):
        """目标帧率滑块回调"""
        value = int(self.target_fps_slider.value)
        self.optimization_manager.target_fps = value
        self.target_fps_text.text = str(value)
    
    def update_ui_state(self):
        """更新UI状态以反映当前设置"""
        self.instanced_rendering_toggle.value = '启用' if self.optimization_manager.use_instanced_rendering else '禁用'
        self.mesh_combining_toggle.value = '启用' if self.optimization_manager.use_mesh_combining else '禁用'
        self.distance_culling_toggle.value = '启用' if self.optimization_manager.use_distance_culling else '禁用'
        self.chunk_management_toggle.value = '启用' if self.optimization_manager.use_chunk_management else '禁用'
        self.adaptive_optimization_toggle.value = '启用' if self.optimization_manager.adaptive_optimization else '禁用'
        
        self.render_distance_slider.value = self.optimization_manager.chunk_manager.render_distance
        self.render_distance_text.text = str(int(self.render_distance_slider.value))
        
        self.target_fps_slider.value = self.optimization_manager.target_fps
        self.target_fps_text.text = str(int(self.target_fps_slider.value))
        
        # 更新当前帧率显示
        self.current_fps_text.text = str(int(self.optimization_manager.current_fps))
    
    def update(self):
        """更新UI，应在每帧调用"""
        if self.window.visible:
            self.current_fps_text.text = str(int(self.optimization_manager.current_fps))
    
    def toggle_window(self):
        """切换窗口显示状态"""
        self.window.visible = not self.window.visible
        if self.window.visible:
            self.update_ui_state()
    
    def show_window(self):
        """显示窗口"""
        self.window.visible = True
        self.update_ui_state()
    
    def hide_window(self):
        """隐藏窗口"""
        self.window.visible = False

# 使用示例
"""
# 在游戏初始化时创建优化管理器和UI
optimization_manager = OptimizationManager(chunk_size=8, render_distance=2)
optimization_ui = OptimizationSettingsUI(optimization_manager)

# 在创建方块时添加到优化管理器
for block in blocks:
    optimization_manager.add_block(block)

# 在游戏循环中更新优化管理器和UI
def update():
    # 更新优化管理器
    optimization_manager.update(player, all_blocks)
    
    # 更新UI
    optimization_ui.update()
    
    # 其他游戏逻辑...
"""