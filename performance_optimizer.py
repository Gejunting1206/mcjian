# 性能优化管理器
# 整合视锥体剔除和LOD系统，提供统一的性能优化接口
# 引入算法级优化，包括空间分区、八叉树结构和异步处理

from frustum_culling import frustum_culling_manager
from lod_system import lod_manager
import time
import threading
import numpy as np
from collections import defaultdict
import math

# 八叉树节点类，用于空间分区优化
class OctreeNode:
    """八叉树节点，用于高效空间分区"""
    
    def __init__(self, center, size, max_depth=4, depth=0):
        self.center = center  # 节点中心点
        self.size = size      # 节点大小
        self.max_depth = max_depth  # 最大深度
        self.depth = depth    # 当前深度
        self.children = [None] * 8  # 8个子节点
        self.objects = []     # 当前节点包含的对象
        self.is_leaf = True   # 是否为叶节点
        
        # 计算节点边界
        half_size = size / 2
        self.bounds = {
            'min': (center[0] - half_size, center[1] - half_size, center[2] - half_size),
            'max': (center[0] + half_size, center[1] + half_size, center[2] + half_size)
        }
    
    def insert(self, obj, position):
        """将对象插入八叉树"""
        # 检查对象是否在当前节点范围内
        if not self._is_in_bounds(position):
            return False
        
        # 如果是叶节点且深度未达到最大，且对象数量超过阈值，则分裂
        if self.is_leaf and self.depth < self.max_depth and len(self.objects) >= 8:
            self._split()
        
        # 如果已分裂，尝试将对象插入到子节点
        if not self.is_leaf:
            index = self._get_child_index(position)
            if self.children[index].insert(obj, position):
                return True
        
        # 如果无法插入子节点或是叶节点，则插入当前节点
        self.objects.append((obj, position))
        return True
    
    def _split(self):
        """分裂当前节点为8个子节点"""
        self.is_leaf = False
        quarter_size = self.size / 4
        half_size = self.size / 2
        
        # 创建8个子节点
        for i in range(8):
            # 计算子节点中心点
            x_offset = quarter_size if (i & 1) else -quarter_size
            y_offset = quarter_size if (i & 2) else -quarter_size
            z_offset = quarter_size if (i & 4) else -quarter_size
            
            child_center = (
                self.center[0] + x_offset,
                self.center[1] + y_offset,
                self.center[2] + z_offset
            )
            
            # 创建子节点
            self.children[i] = OctreeNode(child_center, half_size, self.max_depth, self.depth + 1)
        
        # 重新分配当前节点的对象到子节点
        objects_to_redistribute = self.objects
        self.objects = []
        
        for obj, pos in objects_to_redistribute:
            index = self._get_child_index(pos)
            if not self.children[index].insert(obj, pos):
                self.objects.append((obj, pos))  # 如果无法插入子节点，保留在当前节点
    
    def _get_child_index(self, position):
        """获取位置所在的子节点索引"""
        index = 0
        if position[0] >= self.center[0]: index |= 1
        if position[1] >= self.center[1]: index |= 2
        if position[2] >= self.center[2]: index |= 4
        return index
    
    def _is_in_bounds(self, position):
        """检查位置是否在节点范围内"""
        return (self.bounds['min'][0] <= position[0] <= self.bounds['max'][0] and
                self.bounds['min'][1] <= position[1] <= self.bounds['max'][1] and
                self.bounds['min'][2] <= position[2] <= self.bounds['max'][2])
    
    def query_range(self, query_bounds):
        """查询范围内的所有对象"""
        result = []
        
        # 检查当前节点是否与查询范围相交
        if not self._intersects_bounds(query_bounds):
            return result
        
        # 检查当前节点中的对象
        for obj, pos in self.objects:
            if self._position_in_query_bounds(pos, query_bounds):
                result.append(obj)
        
        # 如果不是叶节点，递归查询子节点
        if not self.is_leaf:
            for child in self.children:
                if child is not None:
                    result.extend(child.query_range(query_bounds))
        
        return result
    
    def _intersects_bounds(self, query_bounds):
        """检查当前节点是否与查询范围相交"""
        return not (
            self.bounds['max'][0] < query_bounds['min'][0] or
            self.bounds['min'][0] > query_bounds['max'][0] or
            self.bounds['max'][1] < query_bounds['min'][1] or
            self.bounds['min'][1] > query_bounds['max'][1] or
            self.bounds['max'][2] < query_bounds['min'][2] or
            self.bounds['min'][2] > query_bounds['max'][2]
        )
    
    def _position_in_query_bounds(self, position, query_bounds):
        """检查位置是否在查询范围内"""
        return (query_bounds['min'][0] <= position[0] <= query_bounds['max'][0] and
                query_bounds['min'][1] <= position[1] <= query_bounds['max'][1] and
                query_bounds['min'][2] <= position[2] <= query_bounds['max'][2])


class PerformanceOptimizer:
    """性能优化管理器，整合多种优化技术，提供统一的控制接口"""
    
    def __init__(self):
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.25  # 更新间隔
        
        # 启用/禁用各个优化系统
        self.enable_frustum_culling = True
        self.enable_lod = True
        self.enable_spatial_partitioning = True  # 启用空间分区
        self.enable_multithreading = True       # 启用多线程
        self.enable_occlusion_culling = True    # 启用遮挡剔除
        
        # 自适应性能优化参数
        self.adaptive_mode = True  # 启用自适应模式
        self.target_fps = 30  # 目标帧率
        self.fps_tolerance = 1  # 帧率容差
        self.optimization_level = 2  # 当前优化级别 (0-低, 1-中, 2-高)
        self.last_optimization_change = 0  # 上次调整优化级别的时间
        self.optimization_cooldown = 0.5  # 调整冷却时间
        
        # 渲染距离控制
        self.render_distance_base = 2  # 基础渲染距离
        self.render_distance_min = 1  # 最小渲染距离
        self.render_distance_max = 4  # 最大渲染距离
        
        # 空间分区参数
        self.octree = None  # 八叉树根节点
        self.octree_max_depth = 5  # 八叉树最大深度
        self.spatial_grid_size = 16  # 空间网格大小
        self.spatial_grid = defaultdict(list)  # 空间哈希网格
        
        # 多线程处理
        self.worker_threads = []  # 工作线程池
        self.max_threads = 4      # 最大线程数
        self.thread_tasks = []    # 线程任务队列
        self.thread_lock = threading.Lock()  # 线程锁
        
        # 视锥体优化参数
        self.frustum_cache_enabled = True  # 启用视锥体缓存
        self.frustum_cache = {}  # 视锥体结果缓存
        self.frustum_cache_lifetime = 0.1  # 缓存生命周期(秒)
        
        # 性能统计
        self.stats = {
            'total_blocks': 0,
            'visible_blocks': 0,
            'culled_blocks': 0,
            'high_detail_blocks': 0,
            'medium_detail_blocks': 0,
            'low_detail_blocks': 0,
            'frame_time_ms': 0,
            'current_fps': 0,
            'optimization_level': 2,
            'render_distance': 2,
            'octree_nodes': 0,
            'spatial_cells': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'thread_utilization': 0
        }
        
        # 应用初始设置
        self._apply_settings()
    
    def _apply_settings(self):
        """应用当前设置到各个优化系统"""
        frustum_culling_manager.enabled = self.enable_frustum_culling and self.enabled
        lod_manager.enabled = self.enable_lod and self.enabled
        
        # 初始化空间分区系统
        if self.enable_spatial_partitioning and self.octree is None:
            # 创建一个足够大的八叉树覆盖整个游戏世界
            world_size = self.render_distance_max * 32  # 估计世界大小
            self.octree = OctreeNode((0, 0, 0), world_size, self.octree_max_depth)
            
        # 初始化多线程工作池
        if self.enable_multithreading and not self.worker_threads:
            self._init_thread_pool()
    
    def toggle(self):
        """切换整体优化开关"""
        self.enabled = not self.enabled
        self._apply_settings()
        return self.enabled
    
    def toggle_frustum_culling(self):
        """切换视锥体剔除开关"""
        self.enable_frustum_culling = not self.enable_frustum_culling
        self._apply_settings()
        return self.enable_frustum_culling
    
    def toggle_lod(self):
        """切换LOD系统开关"""
        self.enable_lod = not self.enable_lod
        self._apply_settings()
        return self.enable_lod
    
    def _init_thread_pool(self):
        """初始化多线程工作池"""
        # 清理现有线程
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(0.1)
        
        self.worker_threads = []
        self.thread_tasks = []
        
        # 创建工作线程
        for i in range(self.max_threads):
            thread = threading.Thread(target=self._worker_thread_func, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
    
    def _worker_thread_func(self, thread_id):
        """工作线程函数"""
        while True:
            task = None
            # 获取任务
            with self.thread_lock:
                if self.thread_tasks:
                    task = self.thread_tasks.pop(0)
            
            if task:
                # 执行任务
                func, args, kwargs, callback = task
                try:
                    result = func(*args, **kwargs)
                    if callback:
                        callback(result)
                except Exception as e:
                    print(f"线程任务执行错误: {e}")
            else:
                # 无任务时休眠
                time.sleep(0.01)
    
    def _add_task(self, func, args=(), kwargs={}, callback=None):
        """添加任务到线程池"""
        if self.enable_multithreading:
            with self.thread_lock:
                self.thread_tasks.append((func, args, kwargs, callback))
            return True
        else:
            # 如果未启用多线程，直接执行
            result = func(*args, **kwargs)
            if callback:
                callback(result)
            return result
    
    def _update_spatial_grid(self, blocks):
        """更新空间哈希网格"""
        if not self.enable_spatial_partitioning:
            return blocks
        
        # 清空现有网格
        self.spatial_grid.clear()
        
        # 更新空间哈希网格
        for block in blocks:
            if hasattr(block, 'position'):
                # 计算网格坐标
                grid_x = int(block.position.x // self.spatial_grid_size)
                grid_y = int(block.position.y // self.spatial_grid_size)
                grid_z = int(block.position.z // self.spatial_grid_size)
                grid_key = (grid_x, grid_y, grid_z)
                
                # 添加到网格
                self.spatial_grid[grid_key].append(block)
                
                # 同时更新八叉树
                if self.octree:
                    pos = (block.position.x, block.position.y, block.position.z)
                    self.octree.insert(block, pos)
        
        # 更新统计信息
        self.stats['spatial_cells'] = len(self.spatial_grid)
        
        return blocks
    
    def _get_nearby_cells(self, position, radius):
        """获取附近的空间网格单元"""
        if not self.enable_spatial_partitioning:
            return []
        
        nearby_blocks = []
        center_x = int(position.x // self.spatial_grid_size)
        center_y = int(position.y // self.spatial_grid_size)
        center_z = int(position.z // self.spatial_grid_size)
        
        # 计算网格半径
        grid_radius = max(1, int(radius // self.spatial_grid_size) + 1)
        
        # 遍历附近的网格单元
        for x in range(center_x - grid_radius, center_x + grid_radius + 1):
            for y in range(center_y - grid_radius, center_y + grid_radius + 1):
                for z in range(center_z - grid_radius, center_z + grid_radius + 1):
                    grid_key = (x, y, z)
                    if grid_key in self.spatial_grid:
                        nearby_blocks.extend(self.spatial_grid[grid_key])
        
        return nearby_blocks
    
    def update(self):
        """更新所有优化系统"""
        if not self.enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 使用多线程并行更新各系统
        if self.enable_multithreading:
            # 并行更新视锥体剔除系统
            if self.enable_frustum_culling:
                self._add_task(frustum_culling_manager.update)
            
            # 并行更新LOD系统
            if self.enable_lod:
                self._add_task(lod_manager.update)
            
            # 主线程处理自适应优化和统计
            if self.adaptive_mode:
                self._adaptive_performance_update()
            
            # 清理过期的视锥体缓存
            self._clean_frustum_cache(current_time)
        else:
            # 串行更新视锥体剔除系统
            if self.enable_frustum_culling:
                frustum_culling_manager.update()
            
            # 串行更新LOD系统
            if self.enable_lod:
                lod_manager.update()
            
            # 自适应性能优化
            if self.adaptive_mode:
                self._adaptive_performance_update()
        
        # 更新统计信息
        self._update_stats()
    
    def _clean_frustum_cache(self, current_time):
        """清理过期的视锥体缓存"""
        if not self.frustum_cache_enabled:
            return
        
        # 删除过期的缓存项
        expired_keys = []
        for key, (timestamp, _) in self.frustum_cache.items():
            if current_time - timestamp > self.frustum_cache_lifetime:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.frustum_cache[key]
    
    def _update_stats(self):
        """更新性能统计信息"""
        # 从视锥体剔除系统获取统计信息
        if self.enable_frustum_culling:
            self.stats['total_blocks'] = frustum_culling_manager.stats['total_blocks']
            self.stats['visible_blocks'] = frustum_culling_manager.stats['visible_blocks']
            self.stats['culled_blocks'] = frustum_culling_manager.stats['culled_blocks']
            self.stats['frame_time_ms'] = frustum_culling_manager.stats['frame_time_ms']
        
        # 从LOD系统获取统计信息
        if self.enable_lod:
            self.stats['high_detail_blocks'] = lod_manager.stats['high_detail']
            self.stats['medium_detail_blocks'] = lod_manager.stats['medium_detail']
            self.stats['low_detail_blocks'] = lod_manager.stats['low_detail']
        
        # 获取当前帧率
        from ursina import application
        self.stats['current_fps'] = getattr(application, 'fps', 30)
        
        # 更新其他统计信息
        self.stats['optimization_level'] = self.optimization_level
        self.stats['render_distance'] = self.render_distance_base
    
    def _adaptive_performance_update(self):
        """自适应性能优化，根据当前帧率动态调整优化策略，专注于算法级优化而非简单降低渲染质量"""
        # 获取当前帧率
        from ursina import application
        current_fps = getattr(application, 'fps', 30)
        self.stats['current_fps'] = current_fps
        
        # 检查是否需要调整优化级别
        current_time = time.time()
        if current_time - self.last_optimization_change < self.optimization_cooldown:
            return  # 冷却期内不调整
        
        # 根据帧率与目标帧率的差距决定是否调整优化策略
        fps_diff = current_fps - self.target_fps
        
        # 帧率极度低下，采取算法级紧急优化措施
        if current_fps < 8:
            print(f"性能优化：帧率极度低下 ({current_fps} FPS)，启用算法级紧急优化措施")
            
            # 启用所有优化算法
            self.enable_spatial_partitioning = True
            self.enable_multithreading = True
            self.enable_occlusion_culling = True
            self.frustum_cache_enabled = True
            
            # 优化空间分区参数
            self.spatial_grid_size = 32  # 增大网格尺寸减少网格数量
            self.octree_max_depth = 3   # 减小八叉树深度降低遍历开销
            
            # 优化缓存策略
            self.frustum_cache_lifetime = 0.2  # 增加缓存生命周期
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 设置优化级别
            self.optimization_level = 2
            self._apply_optimization_level()
            self.last_optimization_change = current_time
            
        # 帧率极低，采取算法优化措施
        elif current_fps < 15:
            print(f"性能优化：帧率极低 ({current_fps} FPS)，启用算法优化措施")
            
            # 启用关键优化算法
            self.enable_spatial_partitioning = True
            self.enable_multithreading = True
            self.frustum_cache_enabled = True
            
            # 优化空间分区参数
            self.spatial_grid_size = 24  # 适度增大网格尺寸
            self.octree_max_depth = 4    # 适度调整八叉树深度
            
            # 优化缓存策略
            self.frustum_cache_lifetime = 0.15  # 适度增加缓存生命周期
            
            # 设置优化级别
            self.optimization_level = 2
            self._apply_optimization_level()
            self.last_optimization_change = current_time
            
        # 帧率低于目标，适度优化
        elif fps_diff < -self.fps_tolerance:
            print(f"性能优化：帧率低于目标 ({current_fps} FPS)，适度优化算法参数")
            
            # 启用基本优化算法
            self.enable_spatial_partitioning = True
            self.enable_multithreading = current_fps < 25  # 仅在较低帧率时启用多线程
            self.frustum_cache_enabled = True
            
            # 优化空间分区参数
            self.spatial_grid_size = 20  # 适度调整网格尺寸
            self.octree_max_depth = 4    # 保持适中的八叉树深度
            
            # 优化缓存策略
            self.frustum_cache_lifetime = 0.1  # 适中的缓存生命周期
            
            # 增加优化级别（如果需要）
            if self.optimization_level < 2:
                self.optimization_level += 1
                self._apply_optimization_level()
                self.last_optimization_change = current_time
                print(f"性能优化：增加优化级别至 {self.optimization_level}")
            
        # 帧率充足，可以减少优化
        elif fps_diff > self.fps_tolerance * 2 and current_fps > 40:
            print(f"性能优化：帧率充足 ({current_fps} FPS)，优化视觉质量")
            
            # 保持空间分区优化
            self.enable_spatial_partitioning = True
            
            # 减少多线程使用，降低CPU负担
            self.enable_multithreading = False
            
            # 优化空间分区参数以提高精度
            self.spatial_grid_size = 16  # 减小网格尺寸提高精度
            self.octree_max_depth = 5    # 增加八叉树深度提高精度
            
            # 减少缓存依赖
            self.frustum_cache_lifetime = 0.05  # 减少缓存生命周期提高准确性
            
            # 降低优化级别（如果可能）
            if self.optimization_level > 0:
                self.optimization_level -= 1
                self._apply_optimization_level()
                self.last_optimization_change = current_time
                print(f"性能优化：降低优化级别至 {self.optimization_level}，提高视觉质量")
        
        # 动态调整空间分区参数
        self._adjust_spatial_parameters(current_fps)
        
        # 更新线程池大小
        if self.enable_multithreading:
            import multiprocessing
            optimal_threads = max(2, min(multiprocessing.cpu_count() - 1, 4))
            if current_fps < 20:
                # 低帧率时减少线程数以降低CPU负担
                self.max_threads = max(2, optimal_threads - 1)
            else:
                # 帧率正常时使用最佳线程数
                self.max_threads = optimal_threads
    
    def _adjust_spatial_parameters(self, current_fps):
        """根据帧率动态调整空间分区参数"""
        if not self.enable_spatial_partitioning:
            return
            
        # 根据帧率调整空间网格大小
        if current_fps < 15:
            # 低帧率时增大网格尺寸减少计算量
            target_grid_size = 32
        elif current_fps < 25:
            target_grid_size = 24
        elif current_fps < 40:
            target_grid_size = 20
        else:
            # 高帧率时减小网格尺寸提高精度
            target_grid_size = 16
            
        # 平滑过渡，避免突变
        if self.spatial_grid_size < target_grid_size:
            self.spatial_grid_size = min(target_grid_size, self.spatial_grid_size + 2)
        elif self.spatial_grid_size > target_grid_size:
            self.spatial_grid_size = max(target_grid_size, self.spatial_grid_size - 2)
            
        # 调整八叉树深度
        if current_fps < 15:
            self.octree_max_depth = 3  # 低帧率时减小深度
        elif current_fps > 40:
            self.octree_max_depth = 5  # 高帧率时增加深度提高精度
        else:
            self.octree_max_depth = 4  # 中等帧率使用平衡值
    
    def _apply_optimization_level(self):
        """应用当前优化级别的设置，专注于算法级优化而非简单降低渲染质量"""
        # 根据优化级别调整各项参数
        if self.optimization_level == 0:  # 低优化 - 高视觉质量
            # 保持较高的渲染距离和视觉质量
            self.render_distance_base = min(self.render_distance_max, 4)
            
            # 基础系统参数
            frustum_culling_manager.update_interval = 0.1
            frustum_culling_manager.culling_radius = 64
            lod_manager.update_interval = 0.2
            
            # 算法优化参数 - 高质量模式
            self.enable_spatial_partitioning = True  # 保持空间分区优化
            self.enable_multithreading = False      # 关闭多线程以减少CPU负担
            self.enable_occlusion_culling = False   # 关闭遮挡剔除以提高视觉质量
            self.frustum_cache_enabled = True       # 启用视锥体缓存
            self.frustum_cache_lifetime = 0.05      # 短缓存生命周期提高准确性
            
            # 空间分区参数 - 高精度
            self.spatial_grid_size = 16             # 小网格尺寸提高精度
            self.octree_max_depth = 5               # 深八叉树提高精度
        
        elif self.optimization_level == 1:  # 中等优化 - 平衡
            # 平衡的渲染距离
            self.render_distance_base = 3
            
            # 基础系统参数
            frustum_culling_manager.update_interval = 0.15
            frustum_culling_manager.culling_radius = 48
            lod_manager.update_interval = 0.25
            
            # 算法优化参数 - 平衡模式
            self.enable_spatial_partitioning = True   # 启用空间分区
            self.enable_multithreading = True        # 启用多线程但限制线程数
            self.max_threads = 2                     # 限制线程数以平衡CPU负担
            self.enable_occlusion_culling = True     # 启用遮挡剔除
            self.frustum_cache_enabled = True        # 启用视锥体缓存
            self.frustum_cache_lifetime = 0.1        # 中等缓存生命周期
            
            # 空间分区参数 - 平衡
            self.spatial_grid_size = 20              # 中等网格尺寸
            self.octree_max_depth = 4                # 中等八叉树深度
        
        else:  # 高优化 - 最佳性能
            # 适度的渲染距离
            self.render_distance_base = max(self.render_distance_min, 2)
            
            # 基础系统参数
            frustum_culling_manager.update_interval = 0.25
            frustum_culling_manager.culling_radius = 32
            lod_manager.update_interval = 0.3
            
            # 算法优化参数 - 高性能模式
            self.enable_spatial_partitioning = True   # 启用空间分区
            self.enable_multithreading = True        # 启用多线程
            self.max_threads = 4                     # 最大线程数提高并行性
            self.enable_occlusion_culling = True     # 启用遮挡剔除
            self.frustum_cache_enabled = True        # 启用视锥体缓存
            self.frustum_cache_lifetime = 0.2        # 长缓存生命周期提高性能
            
            # 空间分区参数 - 高性能
            self.spatial_grid_size = 32              # 大网格尺寸减少计算量
            self.octree_max_depth = 3                # 浅八叉树减少遍历开销
            
            # 如果线程池未初始化，初始化线程池
            if self.enable_multithreading and not self.worker_threads:
                self._init_thread_pool()
        
        # 更新全局变量
        global RENDER_DISTANCE
        try:
            import __main__
            if hasattr(__main__, 'RENDER_DISTANCE'):
                __main__.RENDER_DISTANCE = self.render_distance_base
        except:
            pass
    
    def get_visible_blocks(self, spatial_grid, position, radius=1):
        """获取可见的方块，应用所有启用的优化"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if self.frustum_cache_enabled:
            # 创建缓存键（位置和半径的组合）
            pos_key = (round(position.x, 1), round(position.y, 1), round(position.z, 1))
            cache_key = (pos_key, radius)
            
            # 检查缓存中是否有结果
            if cache_key in self.frustum_cache:
                timestamp, cached_blocks = self.frustum_cache[cache_key]
                if time.time() - timestamp <= self.frustum_cache_lifetime:
                    self.stats['cache_hits'] += 1
                    return cached_blocks
        
        self.stats['cache_misses'] += 1
        
        # 使用空间分区获取附近方块
        if self.enable_spatial_partitioning and self.spatial_grid:
            # 使用空间哈希网格快速查找
            nearby_blocks = self._get_nearby_cells(position, radius)
        elif self.enable_spatial_partitioning and self.octree:
            # 使用八叉树查询
            query_bounds = {
                'min': (position.x - radius, position.y - radius, position.z - radius),
                'max': (position.x + radius, position.y + radius, position.z + radius)
            }
            nearby_blocks = self.octree.query_range(query_bounds)
        else:
            # 使用传统方法
            nearby_blocks = spatial_grid.get_nearby_blocks(position, radius)
        
        # 应用视锥体剔除 - 使用多线程加速
        visible_blocks = nearby_blocks
        if self.enable_frustum_culling and self.enabled:
            if self.enable_multithreading:
                # 将方块分成多个批次并行处理
                batch_size = max(1, len(nearby_blocks) // self.max_threads)
                batches = [nearby_blocks[i:i+batch_size] for i in range(0, len(nearby_blocks), batch_size)]
                
                # 存储结果的列表
                results = []
                
                # 定义回调函数来收集结果
                def collect_result(result):
                    results.extend(result)
                
                # 提交批处理任务
                for batch in batches:
                    self._add_task(
                        frustum_culling_manager.filter_visible_entities,
                        args=(batch,),
                        callback=collect_result
                    )
                
                # 等待所有任务完成（简化版，实际应使用更好的同步机制）
                # 这里使用简单的轮询，实际项目中应使用更高效的同步方式
                max_wait = 0.05  # 最大等待时间
                wait_start = time.time()
                while len(results) < len(batches) and time.time() - wait_start < max_wait:
                    time.sleep(0.001)
                
                # 合并结果
                visible_blocks = []
                for batch_result in results:
                    visible_blocks.extend(batch_result)
            else:
                # 单线程处理
                visible_blocks = frustum_culling_manager.filter_visible_entities(nearby_blocks)
        
        # 应用LOD系统
        if self.enable_lod and self.enabled:
            from ursina import player
            if player and hasattr(player, 'position'):
                visible_blocks = lod_manager.process_blocks_lod(visible_blocks, player.position)
        
        # 更新缓存
        if self.frustum_cache_enabled and cache_key is not None:
            self.frustum_cache[cache_key] = (time.time(), visible_blocks)
        
        # 更新性能统计
        self.stats['frame_time_ms'] = (time.time() - start_time) * 1000
        
        return visible_blocks

# 全局性能优化管理器实例
performance_optimizer = PerformanceOptimizer()

# 键盘快捷键处理函数
def handle_optimization_hotkeys(key):
    """处理性能优化相关的键盘快捷键"""
    if key == 'f8':  # F8: 切换所有优化
        enabled = performance_optimizer.toggle()
        print(f"性能优化: {'开启' if enabled else '关闭'}")
    elif key == 'f9':  # F9: 切换视锥体剔除
        enabled = performance_optimizer.toggle_frustum_culling()
        print(f"视锥体剔除: {'开启' if enabled else '关闭'}")
    elif key == 'f10':  # F10: 切换LOD系统
        enabled = performance_optimizer.toggle_lod()
        print(f"LOD系统: {'开启' if enabled else '关闭'}")
    elif key == 'f7':  # F7: 切换自适应优化
        performance_optimizer.adaptive_mode = not performance_optimizer.adaptive_mode
        print(f"自适应优化: {'开启' if performance_optimizer.adaptive_mode else '关闭'}")
    elif key == 'f6':  # F6: 手动调整优化级别
        performance_optimizer.optimization_level = (performance_optimizer.optimization_level + 1) % 3
        performance_optimizer._apply_optimization_level()
        level_names = ['低(高质量)', '中(平衡)', '高(高性能)']
        print(f"优化级别: {level_names[performance_optimizer.optimization_level]}")