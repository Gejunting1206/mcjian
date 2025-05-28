# GPU优化集成模块
# 将GPU加速渲染、高级着色器优化和GPU视锥体剔除整合到现有系统中

from ursina import *
import time
import math
from collections import deque

# 导入现有优化系统
try:
    from performance_optimizer import performance_optimizer
except ImportError:
    print("警告: 性能优化器未找到")
    performance_optimizer = None

try:
    from frustum_culling import frustum_culling_manager
except ImportError:
    print("警告: 视锥体剔除管理器未找到")
    frustum_culling_manager = None

try:
    from instanced_rendering import InstancedRenderer
except ImportError:
    print("警告: 实例化渲染器未找到")
    InstancedRenderer = None

# 导入新的GPU优化系统
try:
    from gpu_accelerated_rendering import gpu_renderer
except ImportError:
    print("警告: GPU加速渲染器未找到")
    gpu_renderer = None

try:
    from advanced_shader_optimization import advanced_shader_optimizer
except ImportError:
    print("警告: 高级着色器优化器未找到")
    advanced_shader_optimizer = None

# 导入GPU视锥体剔除
try:
    from gpu_frustum_culling import gpu_frustum_culling
except ImportError:
    print("警告: GPU视锥体剔除模块未找到")
    gpu_frustum_culling = None

class GPUOptimizationIntegrator:
    """GPU优化集成器 - 整合所有GPU加速技术并与现有系统集成"""
    
    def __init__(self):
        # 基础参数
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒）
        
        # 引用现有优化系统
        self.performance_optimizer = performance_optimizer
        self.frustum_culling = frustum_culling_manager
        self.instanced_renderer = InstancedRenderer() if InstancedRenderer else None
        
        # 引用新的GPU优化系统
        self.gpu_renderer = gpu_renderer
        self.shader_optimizer = advanced_shader_optimizer
        self.gpu_frustum_culling = gpu_frustum_culling
        
        # 优化开关
        self.use_gpu_instancing = True       # 使用GPU实例化渲染
        self.use_shader_optimization = True  # 使用着色器优化
        self.use_compute_shaders = False     # 使用计算着色器（高级功能，需要GPU支持）
        self.use_geometry_instancing = True  # 使用几何实例化
        self.use_gpu_frustum_culling = True  # 使用GPU视锥体剔除
        
        # 性能监控
        self.fps_history = deque(maxlen=30)  # 帧率历史记录
        self.frame_time_history = deque(maxlen=30)  # 帧时间历史记录
        self.target_fps = 60                 # 目标帧率
        self.min_acceptable_fps = 30         # 最低可接受帧率
        
        # 自适应优化参数
        self.adaptive_mode = True            # 启用自适应模式
        self.optimization_level = 2          # 优化级别 (0-5，越高性能越好，视觉质量越低)
        self.last_optimization_change = 0    # 上次优化级别变更时间
        self.optimization_cooldown = 0.5     # 优化级别变更冷却时间
        
        # 性能统计
        self.stats = {
            'fps': 0,
            'frame_time_ms': 0,
            'optimization_level': 0,
            'draw_calls': 0,
            'instanced_entities': 0,
            'shader_optimized_entities': 0,
            'gpu_render_time_ms': 0
        }
        
        # 初始化
        self._init_systems()
    
    def _init_systems(self):
        """初始化所有子系统"""
        # 配置现有优化系统
        self._configure_existing_systems()
        
        # 初始化GPU渲染器
        if self.use_gpu_instancing and self.gpu_renderer:
            self.gpu_renderer.enabled = True
            self.gpu_renderer.instancing_enabled = True
        
        # 初始化着色器优化器
        if self.use_shader_optimization and self.shader_optimizer:
            self.shader_optimizer.enabled = True
            
        # 初始化GPU视锥体剔除
        if self.use_gpu_frustum_culling and self.gpu_frustum_culling:
            self.gpu_frustum_culling.enabled = True
    
    def _configure_existing_systems(self):
        """配置现有优化系统"""
        # 配置视锥体剔除
        if self.frustum_culling and hasattr(self.frustum_culling, 'update_interval'):
            self.frustum_culling.update_interval = 0.05  # 更频繁更新视锥体
        
        # 配置性能优化器
        if self.performance_optimizer and hasattr(self.performance_optimizer, 'update_interval'):
            self.performance_optimizer.update_interval = 0.05  # 更频繁更新性能优化
        
        # 配置实例化渲染器
        if self.instanced_renderer and hasattr(self.instanced_renderer, 'update_interval'):
            self.instanced_renderer.update_interval = 0.1  # 更频繁更新实例化渲染
    
    def add_entity(self, entity):
        """添加实体到GPU优化系统"""
        if not self.enabled:
            return
        
        # 根据实体类型选择合适的优化
        entity_type = self._determine_entity_type(entity)
        
        # 应用GPU实例化渲染
        if self.use_gpu_instancing and entity_type in ['block', 'static']:
            self.gpu_renderer.add_entity(entity)
        
        # 应用着色器优化
        if self.use_shader_optimization:
            shader_type = self._get_shader_type_for_entity(entity, entity_type)
            if shader_type:
                self.shader_optimizer.apply_shader(entity, shader_type)
    
    def _determine_entity_type(self, entity):
        """确定实体类型，用于选择合适的优化策略"""
        # 检查实体属性和标签
        if hasattr(entity, 'block_type') or hasattr(entity, 'is_block'):
            return 'block'
        
        if hasattr(entity, 'is_terrain') or (hasattr(entity, 'tag') and 'terrain' in entity.tag):
            return 'terrain'
        
        if hasattr(entity, 'is_water') or (hasattr(entity, 'tag') and 'water' in entity.tag):
            return 'water'
        
        if hasattr(entity, 'is_skybox') or (hasattr(entity, 'tag') and 'skybox' in entity.tag):
            return 'skybox'
        
        if hasattr(entity, 'is_static') or not hasattr(entity, 'update'):
            return 'static'
        
        return 'dynamic'
    
    def _get_shader_type_for_entity(self, entity, entity_type):
        """根据实体类型选择合适的着色器"""
        if entity_type == 'block':
            return 'block_optimized'
        
        if entity_type == 'terrain':
            return 'terrain_advanced'
        
        if entity_type == 'water':
            return 'water'
        
        if entity_type == 'skybox':
            return 'skybox'
        
        return None
    
    def remove_entity(self, entity):
        """从GPU优化系统中移除实体"""
        if not self.enabled:
            return
        
        # 从GPU渲染器中移除
        if self.use_gpu_instancing:
            self.gpu_renderer.remove_entity(entity)
        
        # 恢复原始着色器
        if self.use_shader_optimization:
            self.shader_optimizer.restore_entity(entity)
    
    def update(self, player_position=None, camera=None, entities=None, delta_time=0.016):
        """更新GPU优化系统"""
        if not self.enabled:
            return entities
        
        current_time = time.time()
        
        # 更新性能统计
        self._update_performance_stats()
        
        # 自适应优化
        if self.adaptive_mode:
            self._adaptive_optimization(current_time)
        
        # 降低更新频率，减少CPU负担
        if current_time - self.last_update_time < self.update_interval:
            return entities
        
        self.last_update_time = current_time
        
        # 更新GPU视锥体剔除
        if self.use_gpu_frustum_culling and self.gpu_frustum_culling and camera:
            self.gpu_frustum_culling.update(camera)
            
            # 如果提供了实体列表，执行视锥体剔除
            if entities:
                visible_entities = self.gpu_frustum_culling.filter_visible_entities(entities, camera.position if camera else None)
                entities = visible_entities
        
        # 更新GPU渲染器
        if self.use_gpu_instancing and self.gpu_renderer:
            self.gpu_renderer.update()
        
        # 更新着色器优化器
        if self.use_shader_optimization and self.shader_optimizer:
            self.shader_optimizer.update(player_position)
            
        return entities
        
        # 更新统计信息
        self._update_gpu_stats()
    
    def _update_performance_stats(self):
        """更新性能统计"""
        # 获取当前帧率
        if hasattr(application, 'fps'):
            current_fps = application.fps
            self.fps_history.append(current_fps)
            self.stats['fps'] = current_fps
        
        # 计算平均帧时间
        if hasattr(application, 'dt'):
            frame_time = application.dt * 1000  # 转换为毫秒
            self.frame_time_history.append(frame_time)
            self.stats['frame_time_ms'] = frame_time
        
        # 更新其他统计信息
        self.stats['optimization_level'] = self.optimization_level
    
    def _update_gpu_stats(self):
        """更新GPU相关统计信息"""
        # 从GPU渲染器获取统计信息
        if self.use_gpu_instancing and self.gpu_renderer and hasattr(self.gpu_renderer, 'stats'):
            self.stats['draw_calls'] = self.gpu_renderer.stats.get('draw_calls', 0)
            self.stats['instanced_entities'] = self.gpu_renderer.stats.get('instances_rendered', 0)
            self.stats['gpu_render_time_ms'] = self.gpu_renderer.stats.get('render_time_ms', 0)
        
        # 从着色器优化器获取统计信息
        if self.use_shader_optimization and self.shader_optimizer and hasattr(self.shader_optimizer, 'stats'):
            self.stats['shader_optimized_entities'] = self.shader_optimizer.stats.get('optimized_entities', 0)
            
        # 从GPU视锥体剔除获取统计信息
        if self.use_gpu_frustum_culling and self.gpu_frustum_culling and hasattr(self.gpu_frustum_culling, 'stats'):
            self.stats['visible_entities'] = self.gpu_frustum_culling.stats.get('visible_entities', 0)
            self.stats['culled_entities'] = self.gpu_frustum_culling.stats.get('culled_entities', 0)
            self.stats['culling_ratio'] = self.gpu_frustum_culling.stats.get('culling_ratio', 0.0)
            self.stats['culling_time_ms'] = self.gpu_frustum_culling.stats.get('culling_time_ms', 0.0)
    
    def _adaptive_optimization(self, current_time):
        """自适应优化 - 根据性能动态调整优化级别"""
        # 冷却期检查
        if current_time - self.last_optimization_change < self.optimization_cooldown:
            return
        
        # 获取平均帧率
        if not self.fps_history:
            return
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # 根据帧率调整优化级别
        old_level = self.optimization_level
        
        if avg_fps < self.min_acceptable_fps * 0.5:  # 帧率极低
            self.optimization_level = min(5, self.optimization_level + 2)  # 大幅提高优化级别
        elif avg_fps < self.min_acceptable_fps:  # 帧率低于最低可接受值
            self.optimization_level = min(5, self.optimization_level + 1)  # 提高优化级别
        elif avg_fps > self.target_fps * 1.2:  # 帧率远高于目标
            self.optimization_level = max(0, self.optimization_level - 1)  # 降低优化级别
        
        # 如果优化级别变更，应用新设置并记录时间
        if old_level != self.optimization_level:
            self.last_optimization_change = current_time
            self._apply_optimization_level()
    
    def _apply_optimization_level(self):
        """应用当前优化级别的设置"""
        # 级别0: 最高视觉质量，最低性能
        # 级别5: 最低视觉质量，最高性能
        
        if self.optimization_level == 0:
            # 最高视觉质量
            self.use_gpu_instancing = True
            self.use_shader_optimization = True
            self.use_compute_shaders = False
            self.use_geometry_instancing = True
            self.use_gpu_frustum_culling = True
            
            # 配置GPU渲染器
            if self.gpu_renderer:
                self.gpu_renderer.instancing_enabled = True
                self.gpu_renderer.update_interval = 0.1
            
            # 配置着色器优化器
            if self.shader_optimizer:
                self.shader_optimizer.update_interval = 0.1
                
            # 配置GPU视锥体剔除
            if self.gpu_frustum_culling:
                self.gpu_frustum_culling.update_interval = 0.05
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.1
            
        elif self.optimization_level == 1:
            # 高视觉质量
            self.use_gpu_instancing = True
            self.use_shader_optimization = True
            self.use_compute_shaders = False
            self.use_geometry_instancing = True
            self.use_gpu_frustum_culling = True
            
            # 配置GPU渲染器
            if self.gpu_renderer:
                self.gpu_renderer.instancing_enabled = True
                self.gpu_renderer.update_interval = 0.08
            
            # 配置着色器优化器
            if self.shader_optimizer:
                self.shader_optimizer.update_interval = 0.08
                
            # 配置GPU视锥体剔除
            if self.gpu_frustum_culling:
                self.gpu_frustum_culling.update_interval = 0.08
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.15
            
        elif self.optimization_level == 2:
            # 平衡模式
            self.use_gpu_instancing = True
            self.use_shader_optimization = True
            self.use_compute_shaders = False
            self.use_geometry_instancing = True
            self.use_gpu_frustum_culling = True
            
            # 配置GPU渲染器
            if self.gpu_renderer:
                self.gpu_renderer.instancing_enabled = True
                self.gpu_renderer.update_interval = 0.05
            
            # 配置着色器优化器
            if self.shader_optimizer:
                self.shader_optimizer.update_interval = 0.05
                
            # 配置GPU视锥体剔除
            if self.gpu_frustum_culling:
                self.gpu_frustum_culling.update_interval = 0.1
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.2
            
        elif self.optimization_level == 3:
            # 性能优先
            self.use_gpu_instancing = True
            self.use_shader_optimization = True
            self.use_compute_shaders = True
            self.use_geometry_instancing = True
            self.use_gpu_frustum_culling = True
            
            # 配置GPU渲染器
            if self.gpu_renderer:
                self.gpu_renderer.instancing_enabled = True
                self.gpu_renderer.update_interval = 0.1  # 降低更新频率以减少CPU负担
            
            # 配置着色器优化器
            if self.shader_optimizer:
                self.shader_optimizer.update_interval = 0.1
                
            # 配置GPU视锥体剔除
            if self.gpu_frustum_culling:
                self.gpu_frustum_culling.update_interval = 0.15
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.3
                self.gpu_frustum_culling.use_spatial_hash = True
            
        elif self.optimization_level == 4:
            # 高性能模式
            self.use_gpu_instancing = True
            self.use_shader_optimization = True
            self.use_compute_shaders = True
            self.use_geometry_instancing = True
            self.use_gpu_frustum_culling = True
            
            # 配置GPU渲染器
            if self.gpu_renderer:
                self.gpu_renderer.instancing_enabled = True
                self.gpu_renderer.update_interval = 0.2  # 进一步降低更新频率
            
            # 配置着色器优化器
            if self.shader_optimizer:
                self.shader_optimizer.update_interval = 0.2
                
            # 配置GPU视锥体剔除
            if self.gpu_frustum_culling:
                self.gpu_frustum_culling.update_interval = 0.2
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.4
                self.gpu_frustum_culling.use_spatial_hash = True
                self.gpu_frustum_culling.use_octree = True
            
        elif self.optimization_level == 5:
            # 极限性能模式
            self.use_gpu_instancing = True
            self.use_shader_optimization = True
            self.use_compute_shaders = True
            self.use_geometry_instancing = True
            self.use_gpu_frustum_culling = True
            
            # 配置GPU渲染器
            if self.gpu_renderer:
                self.gpu_renderer.instancing_enabled = True
                self.gpu_renderer.update_interval = 0.3  # 最低更新频率
            
            # 配置着色器优化器
            if self.shader_optimizer:
                self.shader_optimizer.update_interval = 0.3
                
            # 配置GPU视锥体剔除
            if self.gpu_frustum_culling:
                self.gpu_frustum_culling.update_interval = 0.3
                self.gpu_frustum_culling.use_frustum_cache = True
                self.gpu_frustum_culling.frustum_cache_lifetime = 0.5
                self.gpu_frustum_culling.use_spatial_hash = True
                self.gpu_frustum_culling.use_octree = True
                self.gpu_frustum_culling.batch_size = 128  # 增大批处理大小
        
        # 应用设置到GPU渲染器
        if self.gpu_renderer:
            self.gpu_renderer.enabled = self.use_gpu_instancing
        
        # 应用设置到着色器优化器
        if self.shader_optimizer:
            self.shader_optimizer.enabled = self.use_shader_optimization
            
        # 应用设置到GPU视锥体剔除
        if self.gpu_frustum_culling:
            self.gpu_frustum_culling.enabled = self.use_gpu_frustum_culling
    
    def toggle_gpu_instancing(self):
        """切换GPU实例化渲染"""
        self.use_gpu_instancing = not self.use_gpu_instancing
        if self.gpu_renderer:
            self.gpu_renderer.enabled = self.use_gpu_instancing
            
            if not self.use_gpu_instancing and hasattr(self.gpu_renderer, 'restore_entities'):
                self.gpu_renderer.restore_entities()
        
        return self.use_gpu_instancing
    
    def toggle_shader_optimization(self):
        """切换着色器优化"""
        self.use_shader_optimization = not self.use_shader_optimization
        if self.shader_optimizer:
            self.shader_optimizer.enabled = self.use_shader_optimization
            
            if not self.use_shader_optimization and hasattr(self.shader_optimizer, 'restore_all'):
                self.shader_optimizer.restore_all()
        
        return self.use_shader_optimization
        
    def toggle_gpu_frustum_culling(self):
        """切换GPU视锥体剔除"""
        self.use_gpu_frustum_culling = not self.use_gpu_frustum_culling
        if self.gpu_frustum_culling:
            self.gpu_frustum_culling.enabled = self.use_gpu_frustum_culling
        
        return self.use_gpu_frustum_culling
    
    def toggle_adaptive_mode(self):
        """切换自适应优化模式"""
        self.adaptive_mode = not self.adaptive_mode
        return self.adaptive_mode
    
    def set_optimization_level(self, level):
        """手动设置优化级别"""
        if 0 <= level <= 5:
            self.optimization_level = level
            self._apply_optimization_level()
            return True
        return False

# 创建全局实例
gpu_optimization_integrator = GPUOptimizationIntegrator()