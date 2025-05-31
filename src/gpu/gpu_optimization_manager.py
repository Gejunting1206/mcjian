# GPU优化管理器
# 整合多种GPU加速技术，提供统一的优化接口

from ursina import *
import time
import sys
import gc
import psutil
import numpy as np
from collections import deque

# 导入GPU优化模块
try:
    from gpu_frustum_culling import gpu_frustum_culling
except ImportError:
    print("警告: GPU视锥体剔除模块未找到")
    gpu_frustum_culling = None

try:
    from gpu_accelerated_rendering import GPUAcceleratedRenderer
except ImportError:
    print("警告: GPU加速渲染模块未找到")
    GPUAcceleratedRenderer = None

try:
    from advanced_shader_optimization import AdvancedShaderOptimizer
except ImportError:
    print("警告: 高级着色器优化模块未找到")
    AdvancedShaderOptimizer = None

class GPUOptimizationManager:
    """GPU优化管理器 - 整合多种GPU加速技术，提供统一的优化接口"""
    
    def __init__(self, app=None):
        self.app = app
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒）
        
        # 优化级别
        self.optimization_level = 0  # 0-5，0为最高视觉质量，5为极限性能模式
        
        # 自适应优化
        self.use_adaptive_optimization = True
        self.target_fps = 60
        self.fps_history = deque(maxlen=10)
        self.last_fps_check_time = 0
        self.fps_check_interval = 1.0  # 每秒检查一次
        
        # 初始化GPU优化模块
        self._init_gpu_modules()
        
        # 统计信息
        self.stats = {
            'fps': 0,
            'frame_time_ms': 0,
            'gpu_memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'memory_usage_mb': 0,
            'visible_entities': 0,
            'culled_entities': 0,
            'culling_ratio': 0.0,
            'instanced_entities': 0,
            'shader_optimized_entities': 0,
            'draw_calls': 0,
            'render_time_ms': 0
        }
        
        # 性能监控
        self.process = psutil.Process()
        self.last_memory_check = 0
        self.memory_check_interval = 5.0  # 每5秒检查一次内存使用情况
    
    def _init_gpu_modules(self):
        """初始化GPU优化模块"""
        # GPU视锥体剔除
        self.use_gpu_frustum_culling = True
        self.gpu_frustum_culling = gpu_frustum_culling if gpu_frustum_culling else None
        
        # GPU加速渲染
        self.use_gpu_accelerated_rendering = True
        self.gpu_renderer = None
        if GPUAcceleratedRenderer:
            self.gpu_renderer = GPUAcceleratedRenderer()
        
        # 高级着色器优化
        self.use_advanced_shader_optimization = True
        self.shader_optimizer = None
        if AdvancedShaderOptimizer:
            self.shader_optimizer = AdvancedShaderOptimizer()
    
    def update(self, dt=None, camera=None, entities=None):
        """更新GPU优化系统"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # 降低更新频率
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 更新性能统计
        self._update_performance_stats()
        
        # 自适应优化
        if self.use_adaptive_optimization and current_time - self.last_fps_check_time > self.fps_check_interval:
            self._adjust_optimization_level()
            self.last_fps_check_time = current_time
        
        # 更新GPU视锥体剔除
        if self.use_gpu_frustum_culling and self.gpu_frustum_culling and camera:
            self.gpu_frustum_culling.update(camera)
            
            # 如果提供了实体列表，执行视锥体剔除
            if entities:
                visible_entities = self.gpu_frustum_culling.filter_visible_entities(entities, camera.position if camera else None)
                
                # 更新统计信息
                self.stats['visible_entities'] = len(visible_entities)
                self.stats['culled_entities'] = len(entities) - len(visible_entities)
                if len(entities) > 0:
                    self.stats['culling_ratio'] = self.stats['culled_entities'] / len(entities)
                
                # 返回可见实体
                entities = visible_entities
        
        # 更新GPU加速渲染
        if self.use_gpu_accelerated_rendering and self.gpu_renderer:
            self.gpu_renderer.update(dt)
            
            # 更新统计信息
            if hasattr(self.gpu_renderer, 'stats'):
                self.stats['instanced_entities'] = self.gpu_renderer.stats.get('instanced_entities', 0)
                self.stats['draw_calls'] = self.gpu_renderer.stats.get('draw_calls', 0)
                self.stats['render_time_ms'] = self.gpu_renderer.stats.get('render_time_ms', 0)
        
        # 更新高级着色器优化
        if self.use_advanced_shader_optimization and self.shader_optimizer:
            self.shader_optimizer.update(dt)
            
            # 更新统计信息
            if hasattr(self.shader_optimizer, 'stats'):
                self.stats['shader_optimized_entities'] = self.shader_optimizer.stats.get('optimized_entities', 0)
        
        return entities
    
    def _update_performance_stats(self):
        """更新性能统计信息"""
        # 更新FPS和帧时间
        if self.app and hasattr(self.app, 'time'):
            dt = self.app.time.dt
            if dt > 0:
                current_fps = 1.0 / dt
                self.fps_history.append(current_fps)
                self.stats['fps'] = sum(self.fps_history) / len(self.fps_history)
                self.stats['frame_time_ms'] = dt * 1000
        
        # 定期更新内存和CPU使用情况
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            try:
                # 更新CPU使用率
                self.stats['cpu_usage_percent'] = self.process.cpu_percent()
                
                # 更新内存使用情况
                memory_info = self.process.memory_info()
                self.stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
                
                # 尝试获取GPU内存使用情况（需要额外库支持，如pynvml）
                # 这里只是占位，实际实现需要根据系统和GPU类型调整
                self.stats['gpu_memory_usage_mb'] = 0
                
                self.last_memory_check = current_time
            except Exception as e:
                print(f"更新性能统计时出错: {e}")
    
    def _adjust_optimization_level(self):
        """根据当前帧率自适应调整优化级别"""
        if not self.fps_history:
            return
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # 根据帧率调整优化级别
        if avg_fps < self.target_fps * 0.5:  # 帧率低于目标的50%
            self.set_optimization_level(min(5, self.optimization_level + 1))
        elif avg_fps < self.target_fps * 0.7:  # 帧率低于目标的70%
            self.set_optimization_level(min(4, self.optimization_level + 1))
        elif avg_fps < self.target_fps * 0.9:  # 帧率低于目标的90%
            self.set_optimization_level(min(3, self.optimization_level + 1))
        elif avg_fps > self.target_fps * 1.5:  # 帧率高于目标的150%
            self.set_optimization_level(max(0, self.optimization_level - 1))
        elif avg_fps > self.target_fps * 1.2:  # 帧率高于目标的120%
            self.set_optimization_level(max(1, self.optimization_level - 1))
    
    def set_optimization_level(self, level):
        """设置优化级别
        
        级别说明：
        0 - 最高视觉质量，最小优化
        1 - 高视觉质量，轻度优化
        2 - 平衡模式，中等优化
        3 - 性能优先，较高优化
        4 - 高性能模式，高度优化
        5 - 极限性能模式，最大优化
        """
        level = max(0, min(5, level))
        if level == self.optimization_level:
            return
        
        self.optimization_level = level
        print(f"GPU优化级别已调整为: {level}")
        
        # 根据优化级别调整各模块参数
        if self.gpu_frustum_culling:
            # 调整视锥体剔除参数
            if level <= 1:  # 高质量模式
                self.gpu_frustum_culling.update_interval = 0.05
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.1
            elif level <= 3:  # 平衡模式
                self.gpu_frustum_culling.update_interval = 0.1
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.2
            else:  # 高性能模式
                self.gpu_frustum_culling.update_interval = 0.2
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.5
        
        if self.gpu_renderer:
            # 调整GPU渲染器参数
            if level <= 1:  # 高质量模式
                self.gpu_renderer.update_interval = 0.03
                self.gpu_renderer.max_instances_per_batch = 1000
            elif level <= 3:  # 平衡模式
                self.gpu_renderer.update_interval = 0.05
                self.gpu_renderer.max_instances_per_batch = 2000
            else:  # 高性能模式
                self.gpu_renderer.update_interval = 0.1
                self.gpu_renderer.max_instances_per_batch = 5000
        
        if self.shader_optimizer:
            # 调整着色器优化器参数
            if level <= 1:  # 高质量模式
                self.shader_optimizer.update_interval = 0.03
                self.shader_optimizer.use_high_quality_shaders = True
            elif level <= 3:  # 平衡模式
                self.shader_optimizer.update_interval = 0.05
                self.shader_optimizer.use_high_quality_shaders = True
            else:  # 高性能模式
                self.shader_optimizer.update_interval = 0.1
                self.shader_optimizer.use_high_quality_shaders = False
    
    def add_entity(self, entity, entity_type=None):
        """添加实体到GPU优化系统"""
        # 添加到GPU渲染器
        if self.use_gpu_accelerated_rendering and self.gpu_renderer:
            self.gpu_renderer.add_entity(entity, entity_type)
        
        # 添加到着色器优化器
        if self.use_advanced_shader_optimization and self.shader_optimizer:
            self.shader_optimizer.add_entity(entity, entity_type)
    
    def remove_entity(self, entity):
        """从GPU优化系统中移除实体"""
        # 从GPU渲染器移除
        if self.use_gpu_accelerated_rendering and self.gpu_renderer:
            self.gpu_renderer.remove_entity(entity)
        
        # 从着色器优化器移除
        if self.use_advanced_shader_optimization and self.shader_optimizer:
            self.shader_optimizer.remove_entity(entity)
    
    def toggle(self):
        """切换GPU优化系统"""
        self.enabled = not self.enabled
        return self.enabled
    
    def toggle_gpu_frustum_culling(self):
        """切换GPU视锥体剔除"""
        if self.gpu_frustum_culling:
            self.use_gpu_frustum_culling = not self.use_gpu_frustum_culling
            return self.use_gpu_frustum_culling
        return False
    
    def toggle_gpu_accelerated_rendering(self):
        """切换GPU加速渲染"""
        if self.gpu_renderer:
            self.use_gpu_accelerated_rendering = not self.use_gpu_accelerated_rendering
            return self.use_gpu_accelerated_rendering
        return False
    
    def toggle_advanced_shader_optimization(self):
        """切换高级着色器优化"""
        if self.shader_optimizer:
            self.use_advanced_shader_optimization = not self.use_advanced_shader_optimization
            return self.use_advanced_shader_optimization
        return False
    
    def toggle_adaptive_optimization(self):
        """切换自适应优化"""
        self.use_adaptive_optimization = not self.use_adaptive_optimization
        return self.use_adaptive_optimization
    
    def set_target_fps(self, fps):
        """设置目标帧率"""
        self.target_fps = max(30, min(144, fps))
    
    def get_stats(self):
        """获取性能统计信息"""
        return self.stats
    
    def print_stats(self):
        """打印性能统计信息"""
        stats = self.get_stats()
        print("===== GPU优化统计信息 =====")
        print(f"FPS: {stats['fps']:.1f}")
        print(f"帧时间: {stats['frame_time_ms']:.2f} ms")
        print(f"CPU使用率: {stats['cpu_usage_percent']:.1f}%")
        print(f"内存使用: {stats['memory_usage_mb']:.1f} MB")
        print(f"GPU内存使用: {stats['gpu_memory_usage_mb']:.1f} MB")
        print(f"可见实体: {stats['visible_entities']}")
        print(f"剔除实体: {stats['culled_entities']}")
        print(f"剔除比例: {stats['culling_ratio']:.2f}")
        print(f"实例化实体: {stats['instanced_entities']}")
        print(f"着色器优化实体: {stats['shader_optimized_entities']}")
        print(f"绘制调用: {stats['draw_calls']}")
        print(f"渲染时间: {stats['render_time_ms']:.2f} ms")
        print(f"优化级别: {self.optimization_level}")
        print("============================")

# 创建全局实例
gpu_optimization_manager = GPUOptimizationManager()