# 区块加载系统
# 提供高效的区块加载、预加载和缓存机制
# 优化区块加载速度，减少游戏卡顿

import time
import threading
import logging
import heapq
from queue import PriorityQueue, Empty
from collections import defaultdict, deque
import numpy as np
from ursina import Vec3

# 区块加载优先级常量
PRIORITY_PLAYER = 0.1      # 玩家所在区块 - 最高优先级
PRIORITY_BELOW = 0.2       # 玩家下方区块 - 次高优先级
PRIORITY_FRONT = 0.5       # 玩家前方区块
PRIORITY_FRONT_FAR = 1.5   # 玩家前方较远区块
PRIORITY_NEAR = 3.0        # 玩家附近区块
PRIORITY_NORMAL = 5.0      # 普通区块
PRIORITY_LOW = 10.0        # 低优先级区块
PRIORITY_SIDE = 4.0        # 玩家侧方区块 (新增)
PRIORITY_BEHIND = 8.0      # 玩家后方区块 (新增)

# 区块加载状态
STATUS_QUEUED = 0          # 已加入队列
STATUS_LOADING = 1         # 正在加载
STATUS_LOADED = 2          # 已加载完成
STATUS_FAILED = 3          # 加载失败

class ChunkLoadingSystem:
    """区块加载系统 - 提供高效的区块加载、预加载和缓存机制"""
    
    def __init__(self, max_concurrent_loads=4, cache_size=60):  # 增加缓存大小从50到60
        # 区块加载队列和状态跟踪
        self.load_queue = PriorityQueue()  # 优先级队列
        self.queued_chunks = set()         # 已加入队列的区块集合
        self.chunk_status = {}             # 区块加载状态字典
        self.loading_chunks = set()        # 正在加载的区块集合
        
        # 区块缓存系统
        self.chunk_cache = {}              # 区块数据缓存
        self.cache_size = cache_size       # 缓存大小
        self.cache_hits = 0                # 缓存命中次数
        self.cache_misses = 0              # 缓存未命中次数
        self.cache_access_times = {}       # 缓存访问时间记录
        self.lru_queue = deque()           # LRU队列，用于缓存淘汰
        
        # 区块预加载系统
        self.preload_spiral = []           # 螺旋预加载序列
        self.preload_distance = 3          # 预加载距离
        self.generate_spiral_sequence()    # 生成螺旋序列
        
        # 异步加载控制
        self.max_concurrent_loads = max_concurrent_loads  # 最大并发加载数
        self.load_lock = threading.Lock()  # 加载锁
        
        # 性能统计
        self.stats = {
            'chunks_loaded': 0,
            'chunks_cached': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_load_time': 0,
            'total_load_time': 0,
            'max_load_time': 0,
            'load_times': []
        }
        
        # 加载历史 - 用于自适应优化
        self.load_history = []             # 加载历史记录
        self.history_size = 100            # 历史记录大小
        
        # 区块生成参数
        self.chunk_size = 16               # 区块大小
        self.world_height = 256            # 世界高度
        
        # 初始化日志
        logging.info("区块加载系统初始化完成")
    
    def generate_spiral_sequence(self):
        """生成螺旋预加载序列 - 优化加载顺序"""
        # 清空现有序列
        self.preload_spiral = []
        
        # 生成螺旋序列
        max_dist = self.preload_distance
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 右、上、左、下
        x, z = 0, 0
        dir_idx = 0
        steps = 1
        step_count = 0
        step_change = 0
        
        # 添加中心点
        self.preload_spiral.append((0, 0))
        
        # 生成螺旋
        while max(abs(x), abs(z)) <= max_dist:
            dx, dz = directions[dir_idx]
            x += dx
            z += dz
            self.preload_spiral.append((x, z))
            
            step_count += 1
            if step_count == steps:
                step_count = 0
                dir_idx = (dir_idx + 1) % 4
                step_change += 1
                if step_change == 2:
                    step_change = 0
                    steps += 1
        
        # 计算每个位置的优先级
        self.preload_priorities = {}
        for i, (dx, dz) in enumerate(self.preload_spiral):
            # 优先级基于螺旋序列索引
            self.preload_priorities[(dx, dz)] = i / len(self.preload_spiral) * PRIORITY_NORMAL
    
    def queue_chunk(self, chunk_pos, priority=PRIORITY_NORMAL, force=False):
        """将区块加入加载队列"""
        # 如果区块已在队列中且不强制重新加载，则跳过
        if not force and chunk_pos in self.queued_chunks:
            return False
        
        # 检查缓存
        if chunk_pos in self.chunk_cache:
            self.cache_hits += 1
            self.stats['cache_hits'] += 1
            # 更新访问时间
            self.cache_access_times[chunk_pos] = time.time()
            # 更新LRU队列
            if chunk_pos in self.lru_queue:
                self.lru_queue.remove(chunk_pos)
            self.lru_queue.append(chunk_pos)
            # 返回缓存的区块数据
            return self.chunk_cache[chunk_pos]
        
        self.cache_misses += 1
        self.stats['cache_misses'] += 1
        
        # 加入队列
        with self.load_lock:
            self.load_queue.put((priority, chunk_pos))
            self.queued_chunks.add(chunk_pos)
            self.chunk_status[chunk_pos] = STATUS_QUEUED
        
        return None
    
    def queue_chunks_around_player(self, player_pos, facing_dir=None, max_chunks=None):
        """根据玩家位置和朝向，智能加载周围区块"""
        # 获取玩家所在区块坐标
        player_chunk_x = int(player_pos.x // self.chunk_size)
        player_chunk_z = int(player_pos.z // self.chunk_size)
        player_chunk = (player_chunk_x, player_chunk_z)
        
        # 加载玩家所在区块 - 最高优先级
        self.queue_chunk(player_chunk, PRIORITY_PLAYER)
        
        # 加载玩家下方区块 - 次高优先级
        # 减少检查深度，只检查最近的下方区块
        for depth in range(1, 12, 4):  # 从1-20减少到1-12
            below_y = player_pos.y - depth
            if below_y > 0:
                below_chunk = (player_chunk_x, player_chunk_z)
                self.queue_chunk(below_chunk, PRIORITY_BELOW)
        
        # 如果有朝向信息，优先加载玩家前方区块
        if facing_dir:
            facing_x = round(facing_dir.x)
            facing_z = round(facing_dir.z)
            if facing_x != 0 or facing_z != 0:
                # 玩家前方区块
                front_chunk = (player_chunk_x + facing_x, player_chunk_z + facing_z)
                self.queue_chunk(front_chunk, PRIORITY_FRONT)
                
                # 玩家前方较远区块
                front_far_chunk = (player_chunk_x + facing_x*2, player_chunk_z + facing_z*2)
                self.queue_chunk(front_far_chunk, PRIORITY_FRONT_FAR)
        
        # 使用螺旋序列加载其他区块，但限制数量
        chunks_added = 2  # 已经添加了玩家所在区块和下方区块
        if facing_dir and (facing_x != 0 or facing_z != 0):
            chunks_added += 2  # 如果添加了前方区块，计数+2
        
        # 如果指定了最大区块数量，限制处理的区块数
        max_to_process = max_chunks if max_chunks is not None else len(self.preload_spiral)
        
        # 只处理螺旋序列中的一部分，减少每次处理的区块数量
        # 优先处理距离玩家较近的区块
        spiral_to_process = self.preload_spiral[:min(len(self.preload_spiral), 8)]  # 最多处理8个螺旋位置
        
        for dx, dz in spiral_to_process:
            # 如果达到最大处理数量，退出循环
            if max_chunks is not None and chunks_added >= max_chunks:
                break
                
            chunk_pos = (player_chunk_x + dx, player_chunk_z + dz)
            # 跳过已加入队列的区块
            if chunk_pos == player_chunk or chunk_pos in self.queued_chunks:
                continue
                
            # 计算优先级 - 基于螺旋序列和玩家朝向
            base_priority = self.preload_priorities.get((dx, dz), PRIORITY_NORMAL)
            
            # 如果有朝向信息，调整优先级
            if facing_dir and (facing_x != 0 or facing_z != 0):
                # 计算区块相对于玩家朝向的角度
                dot_product = dx * facing_x + dz * facing_z
                dist_sq = dx*dx + dz*dz # 距离平方
                
                if dot_product > 0:  # 在玩家前方
                    # 根据距离进一步细化前方优先级，大幅提高前方区块优先级
                    if dist_sq <= 4: # 较近的前方
                        base_priority = PRIORITY_FRONT * 0.5 * (dist_sq / 4.0 + 0.3) # 大幅提高前方近处优先级
                    else: # 较远的前方
                        base_priority = PRIORITY_FRONT * (dist_sq / (self.preload_distance**2) + 0.5) # 提高前方远处优先级
                elif dot_product < 0: # 在玩家后方
                    base_priority = PRIORITY_BEHIND * (1 + dist_sq / (self.preload_distance**2)) # 越远优先级越低
                else: # 在玩家侧方
                    base_priority = PRIORITY_SIDE * (1 + dist_sq / (self.preload_distance**2))
            
            self.queue_chunk(chunk_pos, base_priority)
            chunks_added += 1
    
    def process_queue(self, max_chunks=2, chunk_generator=None):  # 默认值从3降到2
        """处理加载队列，加载指定数量的区块"""
        if not chunk_generator:
            logging.error("未提供区块生成器函数")
            return []
        
        chunks_loaded = []
        chunks_processed = 0
        
        # 记录处理开始时间，用于性能监控
        process_start_time = time.time()
        
        # 限制每次处理的区块数量
        while chunks_processed < max_chunks and not self.load_queue.empty():
            try:
                # 获取优先级最高的区块
                with self.load_lock:
                    if self.load_queue.empty():
                        break
                    priority, chunk_pos = self.load_queue.get_nowait()
                    
                    # 如果区块已经在加载中，跳过
                    if chunk_pos in self.loading_chunks:
                        continue
                    
                    # 标记为正在加载
                    self.loading_chunks.add(chunk_pos)
                    self.chunk_status[chunk_pos] = STATUS_LOADING
                
                # 记录加载开始时间
                start_time = time.time()
                
                # 检查缓存
                if chunk_pos in self.chunk_cache:
                    chunk_data = self.chunk_cache[chunk_pos]
                    self.cache_hits += 1
                    self.stats['cache_hits'] += 1
                    
                    # 更新状态
                    self.chunk_status[chunk_pos] = STATUS_LOADED
                    chunks_loaded.append((chunk_pos, chunk_data))
                else:
                    # 检查处理时间是否已经过长，如果过长则提前退出
                    current_process_time = time.time() - process_start_time
                    if current_process_time > 0.016:  # 16ms，约等于一帧的时间
                        break
                        
                    # 生成区块
                    try:
                        chunk_data = chunk_generator(chunk_pos)
                        
                        # 添加到缓存
                        self._add_to_cache(chunk_pos, chunk_data)
                        
                        # 更新状态
                        self.chunk_status[chunk_pos] = STATUS_LOADED
                        chunks_loaded.append((chunk_pos, chunk_data))
                    except Exception as e:
                        logging.error(f"生成区块 {chunk_pos} 时出错: {e}")
                        self.chunk_status[chunk_pos] = STATUS_FAILED
                
                # 记录加载时间
                load_time = time.time() - start_time
                self._record_load_time(load_time)
                
                # 更新计数
                chunks_processed += 1
                
                # 从队列和加载集合中移除
                with self.load_lock:
                    if chunk_pos in self.queued_chunks:
                        self.queued_chunks.remove(chunk_pos)
                    if chunk_pos in self.loading_chunks:
                        self.loading_chunks.remove(chunk_pos)
                
            except Empty:
                break
            except Exception as e:
                logging.error(f"处理区块队列时出错: {e}")
        
        # 记录总处理时间
        total_process_time = time.time() - process_start_time
        if 'process_times' not in self.stats:
            self.stats['process_times'] = []
        
        # 只保留最近10次的处理时间
        self.stats['process_times'].append(total_process_time)
        if len(self.stats['process_times']) > 10:
            self.stats['process_times'] = self.stats['process_times'][-10:]
        
        return chunks_loaded
    
    def _add_to_cache(self, chunk_pos, chunk_data):
        """添加区块数据到缓存"""
        # 如果缓存已满，移除最久未使用的项
        if len(self.chunk_cache) >= self.cache_size and self.lru_queue:
            while len(self.chunk_cache) >= self.cache_size and self.lru_queue:
                oldest_pos = self.lru_queue.popleft()
                if oldest_pos in self.chunk_cache:
                    del self.chunk_cache[oldest_pos]
                    if oldest_pos in self.cache_access_times:
                        del self.cache_access_times[oldest_pos]
        
        # 添加到缓存
        self.chunk_cache[chunk_pos] = chunk_data
        self.cache_access_times[chunk_pos] = time.time()
        self.lru_queue.append(chunk_pos)
        self.stats['chunks_cached'] = len(self.chunk_cache)
    
    def _record_load_time(self, load_time):
        """记录区块加载时间，用于性能统计"""
        self.stats['chunks_loaded'] += 1
        self.stats['total_load_time'] += load_time
        self.stats['max_load_time'] = max(self.stats['max_load_time'], load_time)
        
        # 保持最近100次加载的时间记录
        self.stats['load_times'].append(load_time)
        if len(self.stats['load_times']) > 100:
            self.stats['load_times'].pop(0)
        
        # 计算平均加载时间
        if self.stats['load_times']:
            self.stats['avg_load_time'] = sum(self.stats['load_times']) / len(self.stats['load_times'])
    
    def get_chunk_from_cache(self, chunk_pos):
        """从缓存中获取区块数据"""
        if chunk_pos in self.chunk_cache:
            # 更新访问时间和LRU队列
            self.cache_access_times[chunk_pos] = time.time()
            if chunk_pos in self.lru_queue:
                self.lru_queue.remove(chunk_pos)
            self.lru_queue.append(chunk_pos)
            
            self.cache_hits += 1
            self.stats['cache_hits'] += 1
            return self.chunk_cache[chunk_pos]
        
        self.cache_misses += 1
        self.stats['cache_misses'] += 1
        return None
    
    def clear_cache(self):
        """清空区块缓存"""
        self.chunk_cache.clear()
        self.cache_access_times.clear()
        self.lru_queue.clear()
        self.stats['chunks_cached'] = 0
    
    def optimize_loading_parameters(self):
        """根据性能统计自适应优化加载参数"""
        # 根据平均加载时间调整并发加载数
        avg_time = self.stats['avg_load_time']
        if avg_time > 0:
            if avg_time < 0.01:  # 加载非常快
                self.max_concurrent_loads = min(8, self.max_concurrent_loads + 1)
            elif avg_time > 0.05:  # 加载较慢
                self.max_concurrent_loads = max(2, self.max_concurrent_loads - 1)
        
        # 根据缓存命中率调整缓存大小
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 100:
            hit_rate = self.cache_hits / total_accesses
            if hit_rate < 0.5 and len(self.chunk_cache) >= self.cache_size * 0.9:
                # 缓存命中率低且缓存接近满，增加缓存大小
                self.cache_size = min(200, int(self.cache_size * 1.2))
            elif hit_rate > 0.8 and len(self.chunk_cache) < self.cache_size * 0.5:
                # 缓存命中率高且缓存使用率低，减小缓存大小
                self.cache_size = max(20, int(self.cache_size * 0.8))
    
    def load_chunk(self, chunk_pos):
        """加载单个区块 - 供外部直接调用
        
        Args:
            chunk_pos: 区块坐标
            
        Returns:
            加载的区块数据
        """
        # 检查缓存
        if chunk_pos in self.chunk_cache:
            self.cache_hits += 1
            self.stats['cache_hits'] += 1
            # 更新访问时间和LRU队列
            self.cache_access_times[chunk_pos] = time.time()
            if chunk_pos in self.lru_queue:
                self.lru_queue.remove(chunk_pos)
            self.lru_queue.append(chunk_pos)
            return self.chunk_cache[chunk_pos]
        
        # 标记为正在加载
        with self.load_lock:
            self.loading_chunks.add(chunk_pos)
            self.chunk_status[chunk_pos] = STATUS_LOADING
        
        try:
            # 这里需要调用区块生成器，但我们没有直接的引用
            # 使用默认的区块生成逻辑
            from chunk_loading_optimizer import ChunkLoadingOptimizer
            optimizer = ChunkLoadingOptimizer()
            chunk_data = optimizer._generate_chunk(chunk_pos)
            
            # 添加到缓存
            self._add_to_cache(chunk_pos, chunk_data)
            
            # 更新状态
            self.chunk_status[chunk_pos] = STATUS_LOADED
            
            # 记录加载时间
            self._record_load_time(0.01)  # 使用默认值
            
            return chunk_data
        except Exception as e:
            logging.error(f"直接加载区块 {chunk_pos} 时出错: {e}")
            self.chunk_status[chunk_pos] = STATUS_FAILED
            raise
        finally:
            # 从加载集合中移除
            with self.load_lock:
                if chunk_pos in self.loading_chunks:
                    self.loading_chunks.remove(chunk_pos)
    
    def get_loading_stats(self):
        """获取加载统计信息"""
        return {
            'queue_size': self.load_queue.qsize(),
            'loading_chunks': len(self.loading_chunks),
            'cached_chunks': len(self.chunk_cache),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_load_time': self.stats['avg_load_time'],
            'max_load_time': self.stats['max_load_time'],
            'total_chunks_loaded': self.stats['chunks_loaded']
        }

# 创建全局实例
chunk_loader = ChunkLoadingSystem()

# 辅助函数 - 获取区块坐标
def get_chunk_position(position, chunk_size=16):
    """根据世界坐标获取区块坐标"""
    chunk_x = int(position.x // chunk_size)
    chunk_z = int(position.z // chunk_size)
    return (chunk_x, chunk_z)

# 辅助函数 - 标记区块已加载
def mark_chunk_loaded(chunk_pos):
    """标记区块已加载完成"""
    if chunk_pos in chunk_loader.chunk_status:
        chunk_loader.chunk_status[chunk_pos] = STATUS_LOADED
    if chunk_pos in chunk_loader.loading_chunks:
        chunk_loader.loading_chunks.remove(chunk_pos)
    if chunk_pos in chunk_loader.queued_chunks:
        chunk_loader.queued_chunks.remove(chunk_pos)

# 辅助函数 - 预热缓存
def preload_chunks(player_pos, facing_dir=None, distance=2):
    """预热区块缓存，提前加载玩家可能需要的区块"""
    # 保存原始预加载距离
    original_distance = chunk_loader.preload_distance
    
    # 设置预加载距离
    chunk_loader.preload_distance = distance
    chunk_loader.generate_spiral_sequence()
    
    # 预加载区块
    chunk_loader.queue_chunks_around_player(player_pos, facing_dir)
    
    # 恢复原始预加载距离
    chunk_loader.preload_distance = original_distance
    chunk_loader.generate_spiral_sequence()