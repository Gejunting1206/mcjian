# 区块缓存系统
# 提供高效的区块数据缓存和预计算机制
# 减少区块加载时的计算量和内存占用

import time
import threading
import logging
from collections import defaultdict, deque
import numpy as np

class BlockCache:
    """区块缓存系统 - 提供高效的区块数据缓存和预计算"""
    
    def __init__(self, max_cache_size=100):
        # 导入配置
        from chunk_loading_config import ChunkLoadingConfig
        
        # 缓存容器
        self.block_data_cache = {}      # 区块数据缓存
        self.block_mesh_cache = {}      # 区块网格缓存
        self.block_collision_cache = {} # 区块碰撞数据缓存
        
        # 缓存配置
        cache_dist = ChunkLoadingConfig.get_cache_distribution()
        self.mesh_cache_size = cache_dist['mesh']
        self.data_cache_size = cache_dist['data']
        self.collision_cache_size = cache_dist['collision']
        
        # LRU缓存管理
        self.max_cache_size = max_cache_size  # 最大缓存大小
        self.cache_access_times = {}    # 缓存访问时间记录
        self.cache_access_counts = defaultdict(int)  # 访问次数统计
        self.lru_queue = deque()        # LRU队列，用于缓存淘汰
        
        # 缓存统计
        self.cache_hits = 0             # 缓存命中次数
        self.cache_misses = 0           # 缓存未命中次数
        
        # 线程安全
        self.cache_lock = threading.Lock()
        
        # 性能统计
        self.stats = {
            'data_cache_size': 0,
            'mesh_cache_size': 0,
            'collision_cache_size': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'hit_rate': 0.0,
            'memory_usage': 0
        }
        
        logging.info("区块缓存系统初始化完成")
    
    def get_block_data(self, chunk_pos):
        """获取区块数据，如果缓存中存在则直接返回"""
        with self.cache_lock:
            if chunk_pos in self.block_data_cache:
                self._update_cache_access(chunk_pos)
                self.cache_hits += 1
                self.stats['cache_hits'] += 1
                return self.block_data_cache[chunk_pos]
            
            self.cache_misses += 1
            self.stats['cache_misses'] += 1
            return None
    
    def get_block_mesh(self, chunk_pos):
        """获取区块网格数据，如果缓存中存在则直接返回"""
        with self.cache_lock:
            if chunk_pos in self.block_mesh_cache:
                self._update_cache_access(chunk_pos)
                self.cache_hits += 1
                self.stats['cache_hits'] += 1
                return self.block_mesh_cache[chunk_pos]
            
            self.cache_misses += 1
            self.stats['cache_misses'] += 1
            return None
    
    def get_collision_data(self, chunk_pos):
        """获取区块碰撞数据，如果缓存中存在则直接返回"""
        with self.cache_lock:
            if chunk_pos in self.block_collision_cache:
                self._update_cache_access(chunk_pos)
                self.cache_hits += 1
                self.stats['cache_hits'] += 1
                return self.block_collision_cache[chunk_pos]
            
            self.cache_misses += 1
            self.stats['cache_misses'] += 1
            return None
    
    def cache_block_data(self, chunk_pos, block_data):
        """缓存区块数据"""
        with self.cache_lock:
            # 检查缓存是否已满
            self._check_cache_size()
            
            # 添加到缓存
            self.block_data_cache[chunk_pos] = block_data
            self._update_cache_access(chunk_pos)
            
            # 更新统计信息
            self.stats['data_cache_size'] = len(self.block_data_cache)
            self._update_memory_usage()
    
    def cache_block_mesh(self, chunk_pos, mesh_data):
        """缓存区块网格数据"""
        with self.cache_lock:
            # 检查缓存是否已满
            self._check_cache_size()
            
            # 添加到缓存
            self.block_mesh_cache[chunk_pos] = mesh_data
            self._update_cache_access(chunk_pos)
            
            # 更新统计信息
            self.stats['mesh_cache_size'] = len(self.block_mesh_cache)
            self._update_memory_usage()
    
    def cache_collision_data(self, chunk_pos, collision_data):
        """缓存区块碰撞数据"""
        with self.cache_lock:
            # 检查缓存是否已满
            self._check_cache_size()
            
            # 添加到缓存
            self.block_collision_cache[chunk_pos] = collision_data
            self._update_cache_access(chunk_pos)
            
            # 更新统计信息
            self.stats['collision_cache_size'] = len(self.block_collision_cache)
            self._update_memory_usage()
    
    def _update_cache_access(self, chunk_pos):
        """更新缓存访问记录"""
        # 更新访问时间
        self.cache_access_times[chunk_pos] = time.time()
        
        # 更新LRU队列
        if chunk_pos in self.lru_queue:
            self.lru_queue.remove(chunk_pos)
        self.lru_queue.append(chunk_pos)
    
    def _check_cache_size(self):
        """检查缓存大小，使用改进的缓存淘汰策略"""
        from chunk_loading_config import ChunkLoadingConfig
        
        # 分别检查各类型缓存
        if len(self.block_mesh_cache) > self.mesh_cache_size:
            self._evict_cache_items('mesh')
        if len(self.block_data_cache) > self.data_cache_size:
            self._evict_cache_items('data')
        if len(self.block_collision_cache) > self.collision_cache_size:
            self._evict_cache_items('collision')
            
    def _evict_cache_items(self, cache_type):
        """根据综合评分淘汰缓存项"""
        from chunk_loading_config import ChunkLoadingConfig
        current_time = time.time()
        
        # 获取目标缓存
        cache_map = {
            'mesh': self.block_mesh_cache,
            'data': self.block_data_cache,
            'collision': self.block_collision_cache
        }[cache_type]
        
        # 计算每个缓存项的评分
        scores = {}
        for pos in cache_map:
            recency_score = 1.0 / (current_time - self.cache_access_times.get(pos, 0) + 1)
            frequency_score = self.cache_access_counts[pos] / (self.cache_hits + 1)
            scores[pos] = (recency_score * ChunkLoadingConfig.RECENCY_WEIGHT + 
                          frequency_score * ChunkLoadingConfig.FREQUENCY_WEIGHT)
        
        # 按评分排序并移除低分项
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        items_to_remove = sorted_items[:len(cache_map) // 4]  # 每次移除1/4的缓存
        
        # 移除选中的项
        for pos, _ in items_to_remove:
            if pos in cache_map:
                del cache_map[pos]
            if pos in self.cache_access_times:
                del self.cache_access_times[pos]
            if pos in self.cache_access_counts:
                del self.cache_access_counts[pos]
            if pos in self.lru_queue:
                self.lru_queue.remove(pos)
    
    def clear_cache(self):
        """清空所有缓存"""
        with self.cache_lock:
            self.block_data_cache.clear()
            self.block_mesh_cache.clear()
            self.block_collision_cache.clear()
            self.cache_access_times.clear()
            self.lru_queue.clear()
            
            # 更新统计信息
            self.stats['data_cache_size'] = 0
            self.stats['mesh_cache_size'] = 0
            self.stats['collision_cache_size'] = 0
            self.stats['memory_usage'] = 0
    
    def remove_from_cache(self, chunk_pos):
        """从缓存中移除指定区块的数据"""
        with self.cache_lock:
            # 从各缓存中移除
            if chunk_pos in self.block_data_cache:
                del self.block_data_cache[chunk_pos]
            if chunk_pos in self.block_mesh_cache:
                del self.block_mesh_cache[chunk_pos]
            if chunk_pos in self.block_collision_cache:
                del self.block_collision_cache[chunk_pos]
            if chunk_pos in self.cache_access_times:
                del self.cache_access_times[chunk_pos]
            if chunk_pos in self.lru_queue:
                self.lru_queue.remove(chunk_pos)
            
            # 更新统计信息
            self.stats['data_cache_size'] = len(self.block_data_cache)
            self.stats['mesh_cache_size'] = len(self.block_mesh_cache)
            self.stats['collision_cache_size'] = len(self.block_collision_cache)
            self._update_memory_usage()
    
    def _update_memory_usage(self):
        """估算缓存内存使用量"""
        # 简单估算，实际内存使用量会有所不同
        data_size = len(self.block_data_cache) * 1024  # 假设每个区块数据平均1KB
        mesh_size = len(self.block_mesh_cache) * 5120  # 假设每个网格数据平均5KB
        collision_size = len(self.block_collision_cache) * 512  # 假设每个碰撞数据平均0.5KB
        
        self.stats['memory_usage'] = data_size + mesh_size + collision_size
    
    def optimize_cache_size(self):
        """根据命中率自适应调整缓存大小"""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 100:
            hit_rate = self.cache_hits / total_accesses
            self.stats['hit_rate'] = hit_rate
            
            # 根据命中率调整缓存大小
            if hit_rate < 0.5:
                # 命中率低，增加缓存大小
                self.max_cache_size = min(500, int(self.max_cache_size * 1.2))
            elif hit_rate > 0.9 and len(self.lru_queue) < self.max_cache_size * 0.7:
                # 命中率高且缓存未满，适当减小缓存大小
                self.max_cache_size = max(50, int(self.max_cache_size * 0.9))
    
    def precompute_block_data(self, chunk_pos, terrain_generator):
        """预计算区块数据并缓存"""
        # 检查缓存中是否已存在
        if self.get_block_data(chunk_pos) is not None:
            return self.get_block_data(chunk_pos)
        
        try:
            # 使用地形生成器生成区块数据
            block_data = terrain_generator.generate_chunk_data(chunk_pos)
            
            # 缓存生成的数据
            self.cache_block_data(chunk_pos, block_data)
            
            return block_data
        except Exception as e:
            logging.error(f"预计算区块数据时出错: {e}")
            return None
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        with self.cache_lock:
            total_accesses = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
            
            return {
                'data_cache_size': len(self.block_data_cache),
                'mesh_cache_size': len(self.block_mesh_cache),
                'collision_cache_size': len(self.block_collision_cache),
                'total_cache_size': len(self.block_data_cache) + len(self.block_mesh_cache) + len(self.block_collision_cache),
                'max_cache_size': self.max_cache_size,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'memory_usage': self.stats['memory_usage'] / 1024  # KB转换为MB
            }

# 创建全局实例
block_cache = BlockCache()

# 辅助函数 - 预热缓存
def preload_chunk_data(chunk_positions, terrain_generator):
    """预热缓存，提前计算并缓存多个区块的数据"""
    for chunk_pos in chunk_positions:
        block_cache.precompute_block_data(chunk_pos, terrain_generator)

# 辅助函数 - 获取周围区块
def get_surrounding_chunks(center_pos, radius=1):
    """获取中心区块周围的区块坐标(包括中心区块)"""
    surrounding = []
    center_x, center_z = center_pos
    
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            # 包含中心区块
            surrounding.append((center_x + dx, center_z + dz))
    
    return surrounding