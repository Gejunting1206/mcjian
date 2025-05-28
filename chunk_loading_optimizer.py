# 区块加载优化器
# 整合区块加载系统、区块缓存和性能优化器
# 提供高效的区块加载、预加载和缓存机制

import time
import threading
import logging
from queue import PriorityQueue, Empty
from collections import defaultdict, deque
import numpy as np
from ursina import Vec3

# 导入相关系统
from loading_system import chunk_loader, get_chunk_position, mark_chunk_loaded, preload_chunks
from block_cache import block_cache, preload_chunk_data, get_surrounding_chunks
from performance_optimizer import PerformanceOptimizer

class ChunkLoadingOptimizer:
    """区块加载优化器 - 整合多种优化技术，提供统一的区块加载优化接口"""
    
    def __init__(self):
        # 导入配置
        from chunk_loading_config import ChunkLoadingConfig
        
        # 基础参数
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = ChunkLoadingConfig.UPDATE_INTERVAL  # 使用配置中的更新间隔
        
        # 区块加载参数
        self.max_chunks_per_frame = ChunkLoadingConfig.MAX_CHUNKS_PER_FRAME  # 使用配置中的每帧加载区块数
        self.preload_distance = ChunkLoadingConfig.PRELOAD_DISTANCE      # 预加载距离
        self.unload_distance = ChunkLoadingConfig.UNLOAD_DISTANCE       # 卸载距离
        
        # 异步加载控制
        self.async_loading = True      # 启用异步加载
        self.loading_threads = ChunkLoadingConfig.LOADING_THREADS       # 使用配置中的线程数
        self.thread_pool = None        # 线程池
        
        # 加载优先级参数
        self.prioritize_visible = True  # 优先加载可见区块
        self.prioritize_player_path = True  # 优先加载玩家路径上的区块
        
        # 缓存参数
        self.enable_caching = True     # 启用缓存
        self.cache_size = 100          # 缓存大小
        
        # 性能监控
        self.performance_monitor = PerformanceOptimizer()
        self.adaptive_loading = True   # 自适应加载
        self.fps_history = deque(maxlen=10) # 记录最近10帧的FPS (从20减少到10)
        self.frame_time_history = deque(maxlen=10) # 记录最近10帧的帧时间 (从20减少到10)
        self.target_fps = 60           # 目标FPS
        self.target_frame_time = 1.0 / self.target_fps # 目标帧时间
        
        # 加载状态跟踪
        self.loading_chunks = set()    # 正在加载的区块
        self.loaded_chunks = {}        # 已加载的区块
        self.chunk_states = {}         # 区块状态字典
        self.chunk_last_relevant_time = {} # 记录区块最后一次相关的时间戳
        
        # 性能统计
        self.stats = {
            'chunks_loaded_total': 0,
            'chunks_loaded_last_second': 0,
            'chunks_unloaded_total': 0,
            'load_times': [],
            'avg_load_time': 0,
            'cache_hit_rate': 0,
            'frame_time_impact': 0
        }
        
        # 初始化系统
        self._initialize_systems()
        
        logging.info("区块加载优化器初始化完成")
    
    def _initialize_systems(self):
        """初始化相关系统"""
        # 配置区块加载系统
        chunk_loader.max_concurrent_loads = self.loading_threads
        chunk_loader.cache_size = self.cache_size
        chunk_loader.preload_distance = self.preload_distance
        chunk_loader.generate_spiral_sequence()
        
        # 配置区块缓存系统
        block_cache.max_cache_size = self.cache_size
        
        # 初始化线程池
        if self.async_loading and not self.thread_pool:
            self._init_thread_pool()
    
    def _init_thread_pool(self):
        """初始化线程池"""
        # 使用性能优化器中的线程池
        self.thread_pool = self.performance_monitor.worker_threads
    
    def update(self, player_position, player_direction=None, delta_time=0.016):
        """更新区块加载优化器"""
        if not self.enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        start_time = time.time()
        
        # 获取玩家所在区块
        player_chunk = get_chunk_position(player_position)
        
        # 1. 优先加载玩家下方区块 - 防止掉落
        self._ensure_below_chunks_loaded(player_position)
        
        # 2. 根据玩家位置和朝向，智能加载周围区块
        # 减少每次处理的区块数量，只处理最近的区块
        chunk_loader.queue_chunks_around_player(player_position, player_direction, max_chunks=5) # 限制为5个区块
        
        # 3. 处理加载队列，加载区块
        self._process_loading_queue(player_chunk)
        
        # 4. 卸载远处区块
        # 每次只处理一部分区块，避免卡顿
        if current_time % 3 < 0.1:  # 每3秒左右执行一次完整的卸载
            self._unload_distant_chunks(player_chunk, player_direction)
        
        # 5. 更新性能统计
        self._update_stats(time.time() - start_time)
        
        # 6. 自适应调整加载参数
        if self.adaptive_loading:
            self._adapt_loading_parameters()
    
    def _ensure_below_chunks_loaded(self, player_position):
        """确保玩家下方区块已加载 - 防止掉落"""
        # 检查玩家下方多个深度的区块，但减少检查深度
        for depth in range(1, 12, 4):  # 从1-20减少到1-12
            below_pos = Vec3(player_position.x, player_position.y - depth, player_position.z)
            below_chunk = get_chunk_position(below_pos)
            
            # 如果下方区块未加载，立即加载
            if below_chunk not in self.loaded_chunks and below_chunk not in self.loading_chunks:
                # 使用最高优先级加载
                chunk_loader.queue_chunk(below_chunk, priority=0.1, force=True)
                self.loading_chunks.add(below_chunk)
    
    def _process_loading_queue(self, player_chunk):
        """处理加载队列，加载区块"""
        # 确定本帧要加载的区块数量
        # 根据性能动态调整每帧加载的区块数
        chunks_to_load = self._get_adaptive_chunks_per_frame()
        
        # 处理加载队列
        loaded_chunks = chunk_loader.process_queue(max_chunks=chunks_to_load, chunk_generator=self._generate_chunk)
        
        # 更新已加载区块
        for chunk_pos, chunk_data in loaded_chunks:
            self.loaded_chunks[chunk_pos] = chunk_data
            if chunk_pos in self.loading_chunks:
                self.loading_chunks.remove(chunk_pos)
            
            # 更新统计信息
            self.stats['chunks_loaded_total'] += 1
            self.stats['chunks_loaded_last_second'] += 1
    
    def _get_adaptive_chunks_per_frame(self):
        """根据当前性能动态获取每帧应加载的区块数 - 更保守的加载策略"""
        # 导入配置
        from chunk_loading_config import ChunkLoadingConfig
        base_max_chunks = ChunkLoadingConfig.MAX_CHUNKS_PER_FRAME

        if not self.adaptive_loading or not self.frame_time_history:
            # 如果禁用自适应或没有历史记录，使用配置中的基础值
            return base_max_chunks
        
        # 使用更平滑的平均帧时间
        avg_frame_time = np.mean(list(self.frame_time_history))
        
        # 如果帧时间远低于目标，可以尝试适度增加加载量
        if avg_frame_time < self.target_frame_time * 0.7:
            # 更保守地增加
            adaptive_max = min(base_max_chunks + 2, 12) # 增加量更小，上限降低到12
            return adaptive_max
        elif avg_frame_time < self.target_frame_time * 0.8:
            # 轻微增加
            adaptive_max = min(base_max_chunks + 1, 10) # 增加量更小，上限降低到10
            return adaptive_max
        # 如果帧时间高于目标，大幅减少加载量
        elif avg_frame_time > self.target_frame_time * 1.1:
            # 大幅减少，确保至少加载1个
            adaptive_max = max(base_max_chunks - 2, 1) # 减少量更大
            return adaptive_max
        elif avg_frame_time > self.target_frame_time * 1.05:
            # 适度减少
            adaptive_max = max(base_max_chunks - 1, 1)
            return adaptive_max
        
        # 性能稳定，使用配置中的基础值
        return base_max_chunks

    def _generate_chunk(self, chunk_pos):
        """生成区块 - 这里需要与游戏的区块生成系统集成"""
        # 首先检查缓存
        cached_data = block_cache.get_block_data(chunk_pos)
        if cached_data is not None:
            return cached_data
        
        # 这里应该调用游戏的区块生成函数
        try:
            # 在实际应用中，这里应该调用 Chunk 类的 generate 方法
            from ursina import Chunk
            chunk = Chunk(chunk_pos)
            chunk.generate()
            
            # 检查是否有保存的状态
            from chunk_state_manager import chunk_state_manager
            saved_state = chunk_state_manager.load_chunk_state(chunk_pos)
            if saved_state:
                # 应用保存的状态
                for block_state in saved_state['blocks']:
                    # 找到对应的方块并更新状态
                    for block in chunk.blocks:
                        if block.position == block_state['position']:
                            block.id = block_state['id']
                            break
            
            # 应用玩家的修改
            chunk_state_manager.apply_modifications(chunk_pos, chunk)
            
            # 缓存生成的区块数据
            block_cache.cache_block_data(chunk_pos, chunk)
            
            return chunk
        except Exception as e:
            logging.error(f"生成区块 {chunk_pos} 时出错: {e}")
            return None
    
    def _unload_distant_chunks(self, player_chunk, player_direction=None):
        """卸载远离玩家的区块，考虑内存压力、玩家朝向和区块相关性"""
        from chunk_loading_config import ChunkLoadingConfig
        chunk_keys = list(self.loaded_chunks.keys())
        if not chunk_keys:
            return

        # 动态调整卸载距离和检查数量基于内存压力
        memory_usage = self.performance_monitor.get_memory_usage()
        is_memory_pressure = ChunkLoadingConfig.should_cleanup_memory(memory_usage)
        
        current_unload_distance = self.unload_distance
        check_count = min(10, len(chunk_keys)) # 增加检查数量以更快响应

        if is_memory_pressure:
            current_unload_distance = max(self.preload_distance + 1, self.unload_distance - 1)
            check_count = min(20, len(chunk_keys))
            logging.debug(f"内存压力高，临时卸载距离: {current_unload_distance}, 检查数量: {check_count}")

        # 收集候选卸载区块及其评分
        unload_candidates = []
        current_time = time.time()
        for chunk_pos in chunk_keys:
            dist = self._get_chunk_distance(chunk_pos, player_chunk)
            
            # 基本距离检查
            if dist <= current_unload_distance and not is_memory_pressure:
                continue # 不在卸载范围内且无内存压力
                
            # 计算卸载评分 (分数越低越优先卸载)
            score = 0
            
            # 距离评分 (距离越远分数越低)
            score -= dist * 1.0 
            
            # 时间评分 (越久没用到分数越低)
            last_relevant = self.chunk_last_relevant_time.get(chunk_pos, 0)
            time_since_relevant = current_time - last_relevant
            score -= time_since_relevant * 0.5 # 时间权重稍低
            
            # 方向评分 (后方区块分数降低)
            if player_direction:
                dx = chunk_pos[0] - player_chunk[0]
                dz = chunk_pos[1] - player_chunk[1]
                if dx != 0 or dz != 0: # Avoid division by zero
                    chunk_vector = Vec3(dx, 0, dz).normalized()
                    dot_product = chunk_vector.dot(player_direction.normalized())
                    if dot_product < -0.3: # 在后方
                        score -= 5 # 显著降低后方区块评分

            # 内存压力调整
            if is_memory_pressure and dist > (current_unload_distance * 0.6):
                 score -= 10 # 内存压力大时，距离稍远的区块也显著降低评分
                 
            # 只考虑距离大于预加载范围的，或有内存压力的
            if dist > self.preload_distance or is_memory_pressure:
                 unload_candidates.append((score, chunk_pos))

        # 对候选区块按评分排序 (升序，分数低的在前)
        unload_candidates.sort()

        # 确定要卸载的数量
        num_to_unload = 0
        if is_memory_pressure:
            num_to_unload = min(len(unload_candidates), max(1, check_count // 2)) # 内存压力大时卸载更多
        else:
            # 正常情况下，只卸载评分最低的几个，且距离超过阈值的
            count = 0
            for score, pos in unload_candidates:
                 if self._get_chunk_distance(pos, player_chunk) > current_unload_distance:
                     count += 1
                 if count >= check_count // 3: # 卸载检查数量的三分之一
                     break
            num_to_unload = count
            
        # 执行卸载
        unloaded_count = 0
        for i in range(min(num_to_unload, len(unload_candidates))):
            score, chunk_pos = unload_candidates[i]
            if chunk_pos not in self.loaded_chunks: # 可能已被其他逻辑卸载
                continue
                
            try:
                # 保存状态逻辑 (保持不变)
                chunk_to_save = self.loaded_chunks.get(chunk_pos)
                save_state = False
                if chunk_to_save:
                    if hasattr(chunk_to_save, 'is_modified') and chunk_to_save.is_modified:
                        save_state = True
                if save_state:
                    from chunk_state_manager import chunk_state_manager
                    logging.debug(f"卸载前保存已修改区块 {chunk_pos} (评分: {score:.2f})")
                    chunk_state_manager.save_chunk_state(chunk_pos, chunk_to_save)
                
                # 卸载
                chunk = self.loaded_chunks.pop(chunk_pos, None)
                if chunk:
                    if hasattr(chunk, 'destroy'):
                        chunk.destroy()
                    self.stats['chunks_unloaded_total'] += 1
                    unloaded_count += 1
                    # 从相关性字典中移除
                    self.chunk_last_relevant_time.pop(chunk_pos, None)
                    
            except Exception as e:
                logging.error(f"卸载区块 {chunk_pos} (评分: {score:.2f}) 时出错: {e}")

        # if unloaded_count > 0:
        #     logging.debug(f"本轮卸载 {unloaded_count} 个区块")

    def _get_adaptive_chunks_per_frame(self):
        """根据性能动态调整每帧加载的区块数，平衡帧率和加载速度"""
        from chunk_loading_config import ChunkLoadingConfig
        
        # 更新性能历史记录
        current_fps = self.performance_monitor.get_current_fps()
        current_frame_time = self.performance_monitor.get_frame_time()
        self.fps_history.append(current_fps)
        self.frame_time_history.append(current_frame_time)
        
        # 计算平均性能指标
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 60
        avg_frame_time = sum(self.frame_time_history) / len(self.frame_time_history) if self.frame_time_history else 0.016
        logging.debug(f"自适应加载: Avg FPS={avg_fps:.1f}, Avg FrameTime={avg_frame_time*1000:.1f}ms")

        memory_usage = self.performance_monitor.get_memory_usage()
        
        target_chunks = self.max_chunks_per_frame
        
        # 基于平均FPS动态调整 - 更平滑的策略
        if avg_fps < ChunkLoadingConfig.MIN_FPS_THRESHOLD * 0.9: # FPS较低时
            # 轻微减少加载量，避免急剧下降
            target_chunks = max(1, self.max_chunks_per_frame - 1)
        elif avg_fps > ChunkLoadingConfig.MIN_FPS_THRESHOLD * 1.5: # FPS较高时
            # 允许更快地增加加载量
            target_chunks = min(ChunkLoadingConfig.MAX_CHUNKS_PER_FRAME * 2, self.max_chunks_per_frame + 2)
        elif avg_fps > ChunkLoadingConfig.MIN_FPS_THRESHOLD * 1.1: # FPS良好时
            # 适度增加加载量
            target_chunks = min(ChunkLoadingConfig.MAX_CHUNKS_PER_FRAME, self.max_chunks_per_frame + 1)
        # else: # FPS在阈值附近，保持当前加载量
        #     target_chunks = self.max_chunks_per_frame
        
        # 检查内存使用情况
        if ChunkLoadingConfig.should_cleanup_memory(memory_usage):
            target_chunks = 1  # 内存压力大时最小化加载
        
        # 平滑调整 max_chunks_per_frame
        # 避免突变，逐步趋向目标值
        if target_chunks > self.max_chunks_per_frame:
            self.max_chunks_per_frame += 1
        elif target_chunks < self.max_chunks_per_frame:
            self.max_chunks_per_frame -= 1
        # 限制调整范围
        self.max_chunks_per_frame = max(1, min(ChunkLoadingConfig.MAX_CHUNKS_PER_FRAME * 2, self.max_chunks_per_frame))
        logging.debug(f"自适应加载: 目标区块/帧={target_chunks}, 实际区块/帧={self.max_chunks_per_frame}")

        return self.max_chunks_per_frame
    
    def _update_stats(self, update_time):
        """更新性能统计"""
        # 记录加载时间
        self.stats['load_times'].append(update_time)
        if len(self.stats['load_times']) > 100:
            self.stats['load_times'].pop(0)
        
        # 计算平均加载时间
        if self.stats['load_times']:
            self.stats['avg_load_time'] = sum(self.stats['load_times']) / len(self.stats['load_times'])
        
        # 获取缓存命中率
        cache_stats = block_cache.get_cache_stats()
        self.stats['cache_hit_rate'] = cache_stats['hit_rate']
        
        # 计算对帧时间的影响
        self.stats['frame_time_impact'] = update_time
    
    def _adapt_loading_parameters(self):
        """根据性能指标自适应调整加载参数 - 更平滑的调整策略"""
        from chunk_loading_config import ChunkLoadingConfig
        
        # 使用历史平均性能指标
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 60
        avg_frame_time = sum(self.frame_time_history) / len(self.frame_time_history) if self.frame_time_history else 0.016
        memory_usage = self.performance_monitor.get_memory_usage()
        
        # 根据性能调整预加载距离 (更平滑的调整)
        target_preload_distance = self.preload_distance
        if avg_fps < ChunkLoadingConfig.MIN_FPS_THRESHOLD * 0.85: # 稍微放宽降低阈值
            # FPS较低时才考虑减少预加载距离，且减少幅度较小
            target_preload_distance = max(ChunkLoadingConfig.PRELOAD_DISTANCE // 2, self.preload_distance - 1) # 保证不低于基础值的一半
        elif avg_fps > ChunkLoadingConfig.MIN_FPS_THRESHOLD * 1.3: # 稍微提高增加阈值
            # FPS较高时适度增加预加载距离
            target_preload_distance = min(ChunkLoadingConfig.PRELOAD_DISTANCE + 1, self.preload_distance + 1) # 增加幅度减小
        # 平滑调整
        if target_preload_distance != self.preload_distance:
             self.preload_distance = target_preload_distance
             chunk_loader.preload_distance = self.preload_distance # 更新加载器配置
             chunk_loader.generate_spiral_sequence() # 预加载距离变化，重新生成序列
             logging.info(f"自适应参数: 预加载距离调整为 {self.preload_distance}")

        # 调整更新间隔 (更平滑的调整)
        target_update_interval = self.update_interval
        if avg_frame_time > 0.03: # 对应约 33 FPS，稍微放宽增加间隔的条件
            # 帧时间较高时轻微增加间隔
            target_update_interval = min(ChunkLoadingConfig.UPDATE_INTERVAL * 1.5, self.update_interval * 1.1) # 增加幅度减小
        elif avg_frame_time < 0.018: # 对应约 55 FPS，稍微收紧减少间隔的条件
            # 帧时间较低时适度减少更新间隔
            target_update_interval = max(ChunkLoadingConfig.UPDATE_INTERVAL * 0.7, self.update_interval * 0.9) # 减少幅度减小
        # 平滑调整
        if abs(target_update_interval - self.update_interval) > 0.001: # 调整阈值
            self.update_interval = target_update_interval
            logging.info(f"自适应参数: 更新间隔调整为 {self.update_interval:.3f}s")

        # 内存压力检查
        if ChunkLoadingConfig.should_cleanup_memory(memory_usage):
            logging.warning("内存压力过大，触发清理机制")
            self._trigger_memory_cleanup()
    
    def _trigger_memory_cleanup(self):
        """触发内存清理，在内存压力大时更积极地卸载"""
        # 确保必要的模块已导入
        from chunk_loading_config import ChunkLoadingConfig
        from chunk_state_manager import chunk_state_manager
        from loading_system import get_chunk_position
        from block_cache import block_cache
        import gc
        import logging # Ensure logging is available

        logging.warning("触发内存清理机制")
        
        # 临时采用更小的卸载距离以强制清理
        temporary_unload_distance = max(self.preload_distance + 1, int(self.unload_distance * 0.7))
        logging.debug(f"内存清理：临时卸载距离设置为 {temporary_unload_distance}")

        # 清理比临时距离更远的区块
        try:
            player_chunk = get_chunk_position(self.performance_monitor.get_player_position())
        except Exception as e:
            logging.error(f"内存清理：获取玩家区块位置失败: {e}")
            return
            
        chunks_to_unload = []
        for chunk_pos in list(self.loaded_chunks.keys()): # Iterate over a copy of keys
            try:
                # Assuming _get_chunk_distance exists in this class
                dist = self._get_chunk_distance(chunk_pos, player_chunk) 
                if dist > temporary_unload_distance:
                    chunks_to_unload.append(chunk_pos)
            except Exception as e:
                 logging.error(f"内存清理：计算区块距离 {chunk_pos} 时出错: {e}")
        
        logging.info(f"内存清理：计划卸载 {len(chunks_to_unload)} 个区块")
        unloaded_count = 0
        for chunk_pos in chunks_to_unload:
            if chunk_pos in self.loaded_chunks: # Check if still loaded
                 try:
                    # 保存状态（如果需要且已修改）
                    chunk_to_save = self.loaded_chunks.get(chunk_pos)
                    save_state = False
                    if chunk_to_save:
                        # Use chunk_state_manager to check for modifications
                        if chunk_state_manager.has_modifications(chunk_pos):
                            save_state = True
                            
                    if save_state:
                        logging.debug(f"内存清理：保存已修改区块 {chunk_pos} 的状态")
                        chunk_state_manager.save_chunk_state(chunk_pos, chunk_to_save)
                        # Clear modifications after saving
                        chunk_state_manager.clear_modifications(chunk_pos)

                    # 卸载
                    chunk = self.loaded_chunks.pop(chunk_pos, None)
                    if chunk and hasattr(chunk, 'destroy'):
                        chunk.destroy()
                        self.stats['chunks_unloaded_total'] += 1
                        unloaded_count += 1
                 except Exception as e:
                    logging.error(f"内存清理卸载区块 {chunk_pos} 时出错: {e}")

        if unloaded_count > 0:
             logging.info(f"内存清理：成功卸载 {unloaded_count} 个区块")

        # 强制进行垃圾回收
        gc.collect()
        logging.debug("内存清理：执行垃圾回收")
        
        # 清理缓存 (Assuming optimize_cache_size exists)
        try:
            block_cache.optimize_cache_size()
            logging.debug("内存清理：优化缓存大小")
        except AttributeError:
             logging.warning("内存清理：block_cache 没有 optimize_cache_size 方法")
        except Exception as e:
             logging.error(f"内存清理：优化缓存时出错: {e}")

    def _get_chunk_distance(self, chunk_pos, player_chunk=None):
        """计算区块到玩家的距离 (曼哈顿距离)"""
        if player_chunk is None:
             player_chunk = get_chunk_position(self.performance_monitor.get_player_position())
        dx = chunk_pos[0] - player_chunk[0]
        dz = chunk_pos[1] - player_chunk[1]
        # return max(abs(dx), abs(dz)) # Chebyshev distance
        return abs(dx) + abs(dz) # Manhattan distance

    def toggle(self):
        """切换优化器开关"""
        self.enabled = not self.enabled
        return self.enabled
    
    def clear_all_chunks(self):
        """清空所有已加载区块"""
        # 获取所有已加载区块
        chunk_keys = list(self.loaded_chunks.keys())
        
        # 卸载所有区块
        for chunk_pos in chunk_keys:
            try:
                chunk = self.loaded_chunks.pop(chunk_pos, None)
                if chunk and hasattr(chunk, 'destroy'):
                    chunk.destroy()
            except Exception as e:
                logging.error(f"清空区块 {chunk_pos} 时出错: {e}")
        
        # 清空加载状态
        self.loading_chunks.clear()
        self.chunk_states.clear()
        self.chunk_last_relevant_time.clear() # 清空相关性时间戳
        
        # 清空缓存
        block_cache.clear_cache()
        
        logging.info("已清空所有区块")
    
    def get_loading_stats(self):
        """获取加载统计信息"""
        return {
            'loaded_chunks': len(self.loaded_chunks),
            'loading_chunks': len(self.loading_chunks),
            'chunks_loaded_total': self.stats['chunks_loaded_total'],
            'chunks_unloaded_total': self.stats['chunks_unloaded_total'],
            'avg_load_time': self.stats['avg_load_time'],
            'cache_hit_rate': self.stats['cache_hit_rate'],
            'frame_time_impact': self.stats['frame_time_impact'],
            'max_chunks_per_frame': self.max_chunks_per_frame,
            'preload_distance': self.preload_distance,
            'update_interval': self.update_interval
        }

# 创建全局实例
chunk_loading_optimizer = ChunkLoadingOptimizer()

# 辅助函数 - 预热系统
def preload_initial_chunks(player_position, distance=2):
    """预热系统，提前加载玩家周围的区块"""
    # 获取玩家所在区块
    player_chunk = get_chunk_position(player_position)
    
    # 获取周围区块
    surrounding_chunks = get_surrounding_chunks(player_chunk, distance)
    
    # 预加载区块
    for chunk_pos in surrounding_chunks:
        chunk_loader.queue_chunk(chunk_pos, priority=5.0)
    
    # 处理加载队列
    chunk_loader.process_queue(max_chunks=len(surrounding_chunks), chunk_generator=chunk_loading_optimizer._generate_chunk)
    
    logging.info(f"已预加载 {len(surrounding_chunks)} 个区块")

# 辅助函数 - 集成到主循环
def integrate_with_game_loop(player, delta_time):
    """将区块加载优化器集成到游戏主循环"""
    # 获取玩家位置和朝向
    player_position = player.position
    player_direction = player.forward
    
    # 更新区块加载优化器
    chunk_loading_optimizer.update(player_position, player_direction, delta_time)