# 综合性能优化器
# 整合多种高级优化技术，提供极致性能提升

from ursina import *
import time
import threading
import numpy as np
from collections import defaultdict, deque
from queue import PriorityQueue, Empty
import gc
import math
import sys  # 添加sys模块导入，用于内存优化功能

# 导入现有优化系统
from frustum_culling import frustum_culling_manager
from lod_system import lod_manager
from performance_optimizer import performance_optimizer
from instanced_rendering import InstancedRenderer
from chunk_loading_optimizer import chunk_loading_optimizer

class ComprehensivePerformanceOptimizer:
    """综合性能优化器 - 整合并增强所有现有优化技术，提供极致性能"""
    
    def __init__(self):
        # 基础参数
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.1   # 适当延长更新间隔，减少CPU负担
        
        # 引用现有优化系统
        self.frustum_culling = frustum_culling_manager
        self.lod_system = lod_manager
        self.performance_optimizer = performance_optimizer
        self.instanced_renderer = InstancedRenderer()  # 创建实例化渲染器实例
        self.chunk_loader = chunk_loading_optimizer
        
        # 高级优化开关 - 全部启用并设置为最激进模式
        self.use_gpu_instancing = True       # 使用GPU实例化渲染
        self.use_mesh_batching = True        # 使用网格批处理
        self.use_occlusion_culling = True    # 使用遮挡剔除
        self.use_shader_optimization = True  # 使用着色器优化
        self.use_async_physics = True        # 使用异步物理计算
        self.use_texture_compression = True  # 使用纹理压缩
        self.use_adaptive_resolution = True  # 使用自适应分辨率
        self.use_aggressive_culling = True   # 使用激进剔除
        self.use_minimal_physics = True      # 使用最小物理计算
        self.use_simplified_lighting = True  # 使用简化光照
        
        # 极限优化参数
        self.extreme_mode = True            # 默认启用极限优化模式
        self.extreme_lod_distance = 2        # 极限LOD距离 (从4降低到2)
        self.extreme_render_distance = 0     # 极限渲染距离 (从1降低到0)
        
        # 多线程参数
        self.thread_pool_size = 6            # 进一步减少线程池大小，降低CPU竞争 (从8降低到6)
        self.thread_pool = None              # 线程池
        self.task_queue = deque()            # 任务队列
        self.thread_lock = threading.Lock()  # 线程锁
        
        # 性能监控 - 极限优化
        self.fps_history = deque(maxlen=10)  # 帧率历史记录 (从15减少到10)
        self.frame_time_history = deque(maxlen=10)  # 帧时间历史记录 (从15减少到10)
        self.target_fps = 300                # 极限目标帧率 (从200提高到300)
        self.min_acceptable_fps = 120        # 最低可接受帧率 (从50提高到120)
        
        # 自适应优化参数 - 极限优化
        self.adaptive_mode = True            # 启用自适应模式
        self.optimization_level = 5         # 极限优化级别 (0-5，越高性能越好，视觉质量越低) (修正范围为0-5)
        self.last_optimization_change = 0    # 上次优化级别变更时间
        self.optimization_cooldown = 0.05    # 优化级别变更冷却时间 (从0.1减小到0.05)
        
        # 性能统计
        self.stats = {
            'fps': 0,
            'frame_time_ms': 0,
            'optimization_level': 0,
            'visible_blocks': 0,
            'culled_blocks': 0,
            'instanced_blocks': 0,
            'batched_meshes': 0,
            'draw_calls': 0,
            'memory_usage_mb': 0,
            'gpu_memory_mb': 0,
            'thread_usage': 0
        }
        
        # 初始化
        self._init_systems()
    
    def _init_systems(self):
        """初始化所有子系统"""
        # 初始化线程池
        if not self.thread_pool:
            self._init_thread_pool()
        
        # 配置现有优化系统
        self._configure_existing_systems()
        
        # 初始化GPU实例化渲染
        if self.use_gpu_instancing:
            self._init_gpu_instancing()
    
    def _init_thread_pool(self):
        """初始化线程池"""
        self.thread_pool = []
        for i in range(self.thread_pool_size):
            thread = threading.Thread(target=self._worker_thread, daemon=True)
            thread.start()
            self.thread_pool.append(thread)
    
    def _worker_thread(self):
        """工作线程函数"""
        while True:
            task = None
            # 获取任务
            with self.thread_lock:
                if self.task_queue:
                    task = self.task_queue.popleft()
            
            if task:
                # 执行任务
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"线程任务执行错误: {e}")
            else:
                # 无任务时短暂休眠
                time.sleep(0.0005)  # 使用更短的休眠时间
    
    def _add_task(self, func, *args, **kwargs):
        """添加任务到线程池"""
        with self.thread_lock:
            self.task_queue.append((func, args, kwargs))
    
    def _configure_existing_systems(self):
        """配置现有优化系统"""
        # 配置视锥体剔除
        self.frustum_culling.update_interval = 0.05  # 更频繁更新视锥体
        
        # 配置LOD系统
        self.lod_system.update_interval = 0.1  # 更频繁更新LOD
        
        # 配置性能优化器
        self.performance_optimizer.update_interval = 0.05  # 更频繁更新性能优化
        self.performance_optimizer.adaptive_mode = True  # 启用自适应模式
        
        # 配置实例化渲染器
        if hasattr(self.instanced_renderer, 'update_interval'):
            self.instanced_renderer.update_interval = 0.1  # 更频繁更新实例化渲染
        
        # 配置区块加载器
        self.chunk_loader.update_interval = 0.1  # 更频繁更新区块加载
        self.chunk_loader.async_loading = True  # 启用异步加载
    
    def _init_gpu_instancing(self):
        """初始化GPU实例化渲染"""
        # 这里可以添加更高级的GPU实例化渲染初始化代码
        pass
        
    def register_subsystem(self, name, system):
        """注册优化子系统"""
        # 将子系统存储到字典中
        if not hasattr(self, 'subsystems'):
            self.subsystems = {}
        self.subsystems[name] = system
        
        # 根据系统类型进行特殊处理
        if name == 'frustum_culling':
            self.frustum_culling = system
        elif name == 'lod_system':
            self.lod_system = system
        elif name == 'performance_optimizer':
            self.performance_optimizer = system
        elif name == 'chunk_loading_optimizer':
            self.chunk_loader = system
    
    def update(self, player_position=None, delta_time=0.016):
        """更新综合性能优化器"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # 更新性能统计
        self._update_performance_stats()
        
        # 自适应优化
        if self.adaptive_mode:
            self._adaptive_optimization(current_time)
        
        # 降低更新频率，减少CPU负担
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 多线程更新各子系统
        if player_position:
            # 视锥体剔除更新
            self._add_task(self._update_frustum_culling)
            
            # LOD系统更新
            self._add_task(self._update_lod_system, player_position)
            
            # 区块加载更新
            self._add_task(self._update_chunk_loading, player_position)
        
        # 主线程更新
        self._update_main_thread()
        
        # 垃圾回收
        self._optimize_memory()
    
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
        
        # 获取内存使用情况
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        except ImportError:
            # 如果没有psutil，使用gc模块获取对象数量作为内存使用指标
            self.stats['memory_usage_mb'] = len(gc.get_objects()) / 1000  # 粗略估计
    
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
            self.use_mesh_batching = True
            self.use_occlusion_culling = True
            self.use_shader_optimization = False
            self.use_async_physics = False
            self.use_texture_compression = False
            self.use_adaptive_resolution = False
            self.extreme_mode = False
            
            # 配置LOD系统
            self.lod_system.LOD_LEVELS[0]['distance'] = 12
            self.lod_system.LOD_LEVELS[1]['distance'] = 24
            
            # 配置区块加载
            self.chunk_loader.preload_distance = 1
            self.chunk_loader.max_chunks_per_frame = 2
            
        elif self.optimization_level == 1:
            # 高视觉质量
            self.use_gpu_instancing = True
            self.use_mesh_batching = True
            self.use_occlusion_culling = True
            self.use_shader_optimization = True
            self.use_async_physics = False
            self.use_texture_compression = False
            self.use_adaptive_resolution = False
            self.extreme_mode = False
            
            # 配置LOD系统
            self.lod_system.LOD_LEVELS[0]['distance'] = 10
            self.lod_system.LOD_LEVELS[1]['distance'] = 20
            
            # 配置区块加载
            self.chunk_loader.preload_distance = 2
            self.chunk_loader.max_chunks_per_frame = 3
            
        elif self.optimization_level == 2:
            # 平衡模式
            self.use_gpu_instancing = True
            self.use_mesh_batching = True
            self.use_occlusion_culling = True
            self.use_shader_optimization = True
            self.use_async_physics = True
            self.use_texture_compression = True
            self.use_adaptive_resolution = False
            self.extreme_mode = False
            
            # 配置LOD系统
            self.lod_system.LOD_LEVELS[0]['distance'] = 8
            self.lod_system.LOD_LEVELS[1]['distance'] = 16
            
            # 配置区块加载
            self.chunk_loader.preload_distance = 1
            self.chunk_loader.max_chunks_per_frame = 2
            
        elif self.optimization_level == 3:
            # 性能优先
            self.use_gpu_instancing = True
            self.use_mesh_batching = True
            self.use_occlusion_culling = True
            self.use_shader_optimization = True
            self.use_async_physics = True
            self.use_texture_compression = True
            self.use_adaptive_resolution = True
            self.extreme_mode = False
            
            # 配置LOD系统
            self.lod_system.LOD_LEVELS[0]['distance'] = 6
            self.lod_system.LOD_LEVELS[1]['distance'] = 12
            
            # 配置区块加载
            self.chunk_loader.preload_distance = 1
            self.chunk_loader.max_chunks_per_frame = 1
            
        elif self.optimization_level == 4:
            # 高性能模式
            self.use_gpu_instancing = True
            self.use_mesh_batching = True
            self.use_occlusion_culling = True
            self.use_shader_optimization = True
            self.use_async_physics = True
            self.use_texture_compression = True
            self.use_adaptive_resolution = True
            self.extreme_mode = True
            
            # 配置LOD系统
            self.lod_system.LOD_LEVELS[0]['distance'] = 4
            self.lod_system.LOD_LEVELS[1]['distance'] = 8
            
            # 配置区块加载
            self.chunk_loader.preload_distance = 1
            self.chunk_loader.max_chunks_per_frame = 1
            
        elif self.optimization_level == 5:
            # 极限性能模式
            self.use_gpu_instancing = True
            self.use_mesh_batching = True
            self.use_occlusion_culling = True
            self.use_shader_optimization = True
            self.use_async_physics = True
            self.use_texture_compression = True
            self.use_adaptive_resolution = True
            self.extreme_mode = True
            
            # 配置LOD系统 - 极度降低距离
            self.lod_system.LOD_LEVELS[0]['distance'] = 2  # 从3降低到2
            self.lod_system.LOD_LEVELS[1]['distance'] = 4  # 从6降低到4
            
            # 配置区块加载 - 最小加载量
            self.chunk_loader.preload_distance = 1
            self.chunk_loader.max_chunks_per_frame = 1
            
            # 极限模式特殊设置
            self.extreme_lod_distance = 3  # 从4降低到3
            self.extreme_render_distance = 0  # 进一步降低极限模式下的渲染距离
    
    def _update_frustum_culling(self):
        """更新视锥体剔除"""
        self.frustum_culling.update()
    
    def _update_lod_system(self, player_position):
        """更新LOD系统"""
        self.lod_system.update()
    
    def _update_chunk_loading(self, player_position):
        """更新区块加载"""
        self.chunk_loader.update(player_position)
    
    def _update_main_thread(self):
        """主线程更新"""
        # 更新实例化渲染
        if self.use_gpu_instancing and hasattr(self.instanced_renderer, 'update'):
            self.instanced_renderer.update()
        
        # 更新性能优化器
        self.performance_optimizer.update()
        
        # 应用自适应分辨率
        if self.use_adaptive_resolution:
            self._apply_adaptive_resolution()

    def _apply_adaptive_resolution(self):
        """应用自适应分辨率 - 根据当前帧率动态调整游戏分辨率"""
        if not hasattr(self, 'original_window_size'):
            # 保存原始窗口大小
            self.original_window_size = window.size
            self.current_scale_factor = 1.0
            self.resolution_change_cooldown = 0
        
        # 检查冷却时间
        current_time = time.time()
        if hasattr(self, 'last_resolution_change') and current_time - self.last_resolution_change < 2.0:
            return
        
        # 获取当前帧率
        from ursina import application
        current_fps = getattr(application, 'fps', 30)
        
        # 根据帧率调整分辨率
        new_scale_factor = self.current_scale_factor
        
        if current_fps < self.min_acceptable_fps * 0.7:  # 帧率严重不足
            new_scale_factor = max(0.4, self.current_scale_factor - 0.15)  # 更激进地降低分辨率
        elif current_fps < self.min_acceptable_fps:  # 帧率不足
            new_scale_factor = max(0.4, self.current_scale_factor - 0.1)  # 更激进地降低分辨率
        elif current_fps > self.target_fps * 1.2 and self.current_scale_factor < 1.0:  # 帧率充足
            new_scale_factor = min(1.0, self.current_scale_factor + 0.05)  # 适度提高分辨率
        
        # 如果需要更改分辨率
        if abs(new_scale_factor - self.current_scale_factor) > 0.01:
            self.current_scale_factor = new_scale_factor
            new_size = (int(self.original_window_size[0] * new_scale_factor), 
                        int(self.original_window_size[1] * new_scale_factor))
            window.size = new_size
            self.last_resolution_change = current_time
            print(f"自适应分辨率: 调整为 {new_size[0]}x{new_size[1]} (缩放因子: {new_scale_factor:.2f})")

    def _optimize_memory(self):
        """优化内存使用"""
        # 定期执行垃圾回收
        if not hasattr(self, 'last_gc_time'):
            self.last_gc_time = 0
        
        current_time = time.time()
        if current_time - self.last_gc_time > 3:  # 从5秒减少到3秒，更频繁地执行垃圾回收
            self.last_gc_time = current_time
            # 执行完整的垃圾回收
            gc.collect(2)  # 强制执行完整的垃圾回收（所有代）
            
            # 应用纹理压缩
            if self.use_texture_compression and (not hasattr(self, 'texture_compression_applied') or not self.texture_compression_applied):
                self._apply_texture_compression()
            
            # 清理未使用的缓存
            if hasattr(self, 'chunk_loader') and hasattr(self.chunk_loader, 'clear_unused_cache'):
                self.chunk_loader.clear_unused_cache()
            
            # 清理区块加载器的内存
            if hasattr(self, 'chunk_loading_optimizer') and hasattr(self.chunk_loading_optimizer, '_trigger_memory_cleanup'):
                # 每隔3次垃圾回收，执行一次更彻底的内存清理
                if not hasattr(self, 'memory_cleanup_counter'):
                    self.memory_cleanup_counter = 0
                
                self.memory_cleanup_counter += 1
                if self.memory_cleanup_counter >= 3:
                    self.memory_cleanup_counter = 0
                    self.chunk_loading_optimizer._trigger_memory_cleanup()
            
            # 清理Python内部缓存
            sys.intern.clear() if hasattr(sys, 'intern') and hasattr(sys.intern, 'clear') else None
            
            # 压缩内存（仅在Windows上可用）
            if sys.platform == 'win32':
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                except Exception as e:
                    pass
    
    def _apply_texture_compression(self):
        """应用纹理压缩 - 降低纹理质量以提高性能"""
        try:
            # 获取所有已加载的纹理
            from ursina.texture import Texture
            from ursina.scene import scene
            from ursina.entity import Entity
            
            # 设置全局纹理压缩参数
            Texture.default_filtering = 'bilinear'  # 使用双线性过滤而非三线性
            
            # 遍历场景中的所有实体
            compressed_count = 0
            for entity in scene.entities:
                if hasattr(entity, 'texture') and entity.texture:
                    # 跳过已经压缩的纹理
                    if hasattr(entity.texture, 'compressed') and entity.texture.compressed:
                        continue
                    
                    # 降低纹理分辨率
                    if hasattr(entity.texture, 'width') and entity.texture.width > 64:  # 降低纹理分辨率阈值
                        # 记录原始纹理以便需要时恢复
                        if not hasattr(entity, '_original_texture'):
                            entity._original_texture = entity.texture
                        
                        # 应用压缩 - 降低分辨率并禁用mipmap
                        if hasattr(entity.texture, 'set_format'):
                            entity.texture.set_format('compressed')
                            entity.texture.compressed = True
                            compressed_count += 1
            
            print(f"纹理压缩: 已压缩 {compressed_count} 个纹理")
            self.texture_compression_applied = True
        except Exception as e:
            print(f"纹理压缩失败: {e}")

    def toggle(self):
        """切换优化器开关"""
        self.enabled = not self.enabled
        return self.enabled
    
    def set_optimization_level(self, level):
        """手动设置优化级别"""
        if 0 <= level <= 5:
            self.optimization_level = level
            self._apply_optimization_level()
            return True
        return False
    
    def toggle_extreme_mode(self):
        """切换极限模式"""
        self.extreme_mode = not self.extreme_mode
        return self.extreme_mode

# 创建全局实例
comprehensive_optimizer = ComprehensivePerformanceOptimizer()