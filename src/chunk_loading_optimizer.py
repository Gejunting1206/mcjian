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
        self.max_steps_per_frame = 50  # 每帧最大生成步骤数，控制增量生成负载
        self.preload_distance = ChunkLoadingConfig.PRELOAD_DISTANCE - 1      # 预加载距离减1
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
        self.fps_history = deque(maxlen=30) # 记录最近30帧的FPS (从15增加到30)
        self.frame_time_history = deque(maxlen=30) # 记录最近30帧的帧时间 (从15增加到30)
        self.target_fps = 60          # 目标FPS (从150降低到60)
        self.target_frame_time = 1.0 / self.target_fps # 目标帧时间
        self.aggressive_optimization = True # 启用激进优化模式
        
        # 加载状态跟踪
        self.loading_chunks = set()    # 正在加载的区块
        self.loaded_chunks = {}        # 已加载的区块
        self.chunk_states = {}         # 区块状态字典
        self.chunk_last_relevant_time = {} # 记录区块最后一次相关的时间戳
        
        # 增量生成相关
        self.chunk_generation_data = {}  # 区块生成数据
        self.chunk_generation_progress = {}  # 区块生成进度
        
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

    def _process_incremental_generation(self):
        """处理增量生成，每帧只处理有限数量的步骤"""
        total_steps_processed = 0
        max_steps = self.max_steps_per_frame
        
        # 复制当前需要处理的区块列表，避免在迭代中修改
        chunks_to_process = list(self.chunk_generation_data.keys())
        
        for chunk_pos in chunks_to_process:
            if total_steps_processed >= max_steps:
                break
            
            before_progress = self.chunk_generation_progress.get(chunk_pos, 0)
            chunk_data = self._generate_chunk(chunk_pos)
            after_progress = self.chunk_generation_progress.get(chunk_pos, 0)
            
            steps_processed = after_progress - before_progress
            total_steps_processed += steps_processed
            
            if chunk_data is not None:
                # 区块生成完成，添加到已加载列表
                self.loaded_chunks[chunk_pos] = chunk_data
                self.stats['chunks_loaded_total'] += 1
                # 清理临时数据
                if chunk_pos in self.chunk_generation_progress:
                    del self.chunk_generation_progress[chunk_pos]
                if chunk_pos in self.chunk_generation_data:
                    del self.chunk_generation_data[chunk_pos]

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
        # 获取玩家区块位置并确保为三维元组
        player_chunk = get_chunk_position(player_position)
        if len(player_chunk) == 2:
            player_chunk = (player_chunk[0], 0, player_chunk[1])  # 添加y坐标
        
        # 1. 处理增量生成任务
        self._process_incremental_generation()

        # 2. 优先加载玩家下方区块 - 防止掉落
        self._ensure_below_chunks_loaded(player_position)
        
        # 2. 根据玩家位置和朝向，智能加载周围区块
        # 使用增量生成系统替代旧的chunk_loader
        surrounding_chunks = self._get_surrounding_chunks(player_position, player_direction, max_chunks=3)
        for chunk_pos in surrounding_chunks:
            if chunk_pos not in self.loaded_chunks and chunk_pos not in self.loading_chunks and chunk_pos not in self.chunk_generation_data:
                self.chunk_generation_progress[chunk_pos] = 0
                self.chunk_generation_data[chunk_pos] = {
                    'blocks': [],
                    'heightmap': [],
                    'structures': [],
                    'terrain_generated': False,
                    'structures_generated': False,
                    'mesh_generated': False
                }
                self.loading_chunks.add(chunk_pos)
        
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
    
    def _update_stats(self, update_time):
        """更新性能统计信息"""
        # 更新帧时间影响
        self.stats['frame_time_impact'] = update_time * 1000  # 转换为毫秒
        
        # 计算平均加载时间
        if self.stats['load_times']:
            self.stats['avg_load_time'] = sum(self.stats['load_times']) / len(self.stats['load_times'])
        
        # 计算缓存命中率
        total_requests = block_cache.cache_hits + block_cache.cache_misses
        if total_requests > 0:
            self.stats['cache_hit_rate'] = block_cache.cache_hits / total_requests

    def _ensure_below_chunks_loaded(self, player_position):
        """确保玩家下方区块已加载 - 防止掉落"""
        # 检查玩家下方多个深度的区块，但减少检查深度
        for depth in range(1, 12, 4):  # 从1-20减少到1-12
            below_pos = Vec3(player_position.x, player_position.y - depth, player_position.z)
            below_chunk = get_chunk_position(below_pos)
            # 确保区块位置为三维元组
            if len(below_chunk) == 2:
                below_chunk = (below_chunk[0], 0, below_chunk[1])
            
            # 如果下方区块未加载，立即加载
            if below_chunk not in self.loaded_chunks and below_chunk not in self.loading_chunks and below_chunk not in self.chunk_generation_data:
                # 添加到增量生成队列
                self.chunk_generation_progress[below_chunk] = 0
                self.chunk_generation_data[below_chunk] = {
                    'blocks': [],
                    'heightmap': [],
                    'structures': [],
                    'terrain_generated': False,
                    'structures_generated': False,
                    'mesh_generated': False
                }
                self.loading_chunks.add(below_chunk)

    def _get_surrounding_chunks(self, player_position, direction=None, max_chunks=3):
        """获取玩家周围的区块位置，优先考虑玩家朝向"""
        player_chunk = get_chunk_position(player_position)
        chunks = []
        
        # 添加调试信息
        logging.debug(f"player_chunk type: {type(player_chunk)}, value: {player_chunk}")
        
        try:
            # 生成玩家周围5x5范围内的区块
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if dx == 0 and dz == 0:
                        continue  # 跳过玩家当前所在区块
                    
                    # 根据player_chunk的实际维度使用正确的索引
                    if len(player_chunk) == 2:
                        # 转换为3D区块坐标 (x,y,z)
                        chunk_pos = (
                            player_chunk[0] + dx,
                            0,  # y坐标默认为0
                            player_chunk[1] + dz
                        )
                    elif len(player_chunk) == 3:
                        # 3D区块坐标 (x,y,z)
                        chunk_pos = (
                            player_chunk[0] + dx,
                            player_chunk[1],
                            player_chunk[2] + dz
                        )
                    else:
                        logging.error(f"Unexpected chunk position dimensions: {len(player_chunk)}")
                        continue
                    
                    chunks.append(chunk_pos)
        except IndexError as e:
            logging.error(f"Index error in _get_surrounding_chunks: {e}, player_chunk: {player_chunk}")
            # 使用默认2D坐标作为回退，增加安全检查
            x = player_chunk[0] if len(player_chunk) > 0 else 0
            z = player_chunk[1] if len(player_chunk) > 1 else 0
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if dx == 0 and dz == 0:
                        continue
                    chunk_pos = (x + dx, z + dz)
                    chunks.append(chunk_pos)
        
        # 按距离排序，根据pos维度使用正确的坐标
        chunks.sort(key=lambda pos: (
            (pos[0] - player_chunk[0])**2 + 
            (pos[1] - player_chunk[1])** 2 if len(pos) == 2 else 
            (pos[2] - player_chunk[1])**2
        ))
        
        # 如果提供了方向，优先考虑玩家朝向的区块
        if direction is not None:
            # 简单的方向优先级排序
            dir_x = direction.x
            dir_z = direction.z
            
            def direction_priority(chunk_pos):
                dx = chunk_pos[0] - player_chunk[0]
                dz = chunk_pos[2] - player_chunk[2]
                # 计算方向点积（优先同方向的区块）
                dot_product = dx * dir_x + dz * dir_z
                # 结合距离和方向优先级
                return (-dot_product, (dx**2 + dz**2))
            
            chunks.sort(key=direction_priority)
        
        # 返回最多max_chunks个区块
        return chunks[:max_chunks]
    
    def _unload_distant_chunks(self, player_chunk, player_direction=None):
        """卸载远处区块以释放内存"""
        max_distance = self.preload_distance + 2
        distant_chunks = []
        
        # 找出超出最大距离的区块
        for chunk_pos in self.loaded_chunks.keys():
            distance = self._get_chunk_distance(chunk_pos, player_chunk)
            if distance > max_distance:
                distant_chunks.append(chunk_pos)
        
        # 每次只卸载一部分区块，避免卡顿
        num_to_unload = min(len(distant_chunks), 2)  # 每次最多卸载2个区块
        unloaded_count = 0
        
        for chunk_pos in distant_chunks[:num_to_unload]:
            try:
                chunk = self.loaded_chunks.pop(chunk_pos, None)
                if chunk and hasattr(chunk, 'destroy'):
                    chunk.destroy()
                
                # 更新统计信息
                self.stats['chunks_unloaded_total'] += 1
                unloaded_count += 1
            except Exception as e:
                logging.error(f"卸载区块 {chunk_pos} 时出错: {e}")
        
        if unloaded_count > 0:
            logging.debug(f"已卸载 {unloaded_count} 个远处区块")

    def _process_loading_queue(self, player_chunk):
        """处理加载队列，加载区块"""
        # 确定本帧要加载的区块数量
        # 根据性能动态调整每帧加载的区块数
        chunks_to_load = self._get_adaptive_chunks_per_frame()
        
        # 已迁移到增量生成系统处理
        pass
        
        # 处理加载队列
        # loaded_chunks = chunk_loader.process_queue(max_chunks=chunks_to_load, chunk_generator=self._generate_chunk)
        
        # 更新已加载区块
        # for chunk_pos, chunk_data in loaded_chunks:
        #     self.loaded_chunks[chunk_pos] = chunk_data
        #     if chunk_pos in self.loading_chunks:
        #         self.loading_chunks.remove(chunk_pos)
        #     
        #     # 更新统计信息
        #     self.stats['chunks_loaded_total'] += 1
        #     self.stats['chunks_loaded_last_second'] += 1
    
    def _get_adaptive_chunks_per_frame(self):
        """根据当前性能动态获取每帧应加载的区块数 - 更激进的加载策略"""
        # 导入配置
        from chunk_loading_config import ChunkLoadingConfig
        base_max_chunks = ChunkLoadingConfig.MAX_CHUNKS_PER_FRAME

        if not self.adaptive_loading or not self.frame_time_history:
            # 如果禁用自适应或没有历史记录，使用配置中的基础值
            return base_max_chunks
        
        # 使用更平滑的平均帧时间
        avg_frame_time = np.mean(list(self.frame_time_history))
        
        # 如果帧时间远低于目标，可以大幅增加加载量
        if avg_frame_time < self.target_frame_time * 0.6:
            # 大幅增加
            adaptive_max = min(base_max_chunks + 2, 8) # 降低每帧加载上限，减少卡顿
            return adaptive_max
        elif avg_frame_time < self.target_frame_time * 0.7:
            # 中等增加
            adaptive_max = min(base_max_chunks + 1, 6)
            return adaptive_max
        elif avg_frame_time < self.target_frame_time * 0.85:
            # 小幅增加
            adaptive_max = min(base_max_chunks + 1, 4)
            return adaptive_max
        elif avg_frame_time > self.target_frame_time * 1.5:
            # 大幅减少
            adaptive_max = max(base_max_chunks - 3, 1)
            return adaptive_max
        elif avg_frame_time > self.target_frame_time * 1.2:
            # 中等减少
            adaptive_max = max(base_max_chunks - 2, 2)
            return adaptive_max
        elif avg_frame_time > self.target_frame_time * 1.1:
            # 小幅减少
            adaptive_max = max(base_max_chunks - 1, 3)
            return adaptive_max
        
        # 默认返回基础值
        return base_max_chunks

    def _adapt_loading_parameters(self):
        """根据性能统计自适应调整加载参数"""
        if not self.stats or 'frame_time_impact' not in self.stats:
            return

        # 根据帧时间影响调整更新间隔
        if self.stats['frame_time_impact'] > 30:
            # 如果优化器本身耗时超过30ms，延长更新间隔
            self.update_interval = min(self.update_interval + 0.02, 0.2)
        elif self.stats['frame_time_impact'] < 10 and self.update_interval > 0.05:
            # 如果优化器耗时较少且更新间隔大于0.05，缩短更新间隔
            self.update_interval = max(self.update_interval - 0.01, 0.05)

        # 根据缓存命中率调整预加载距离
        if self.stats.get('cache_hit_rate', 0) > 0.7 and self.preload_distance < 3:
            # 缓存命中率高时增加预加载距离
            self.preload_distance += 0.5
        elif self.stats.get('cache_hit_rate', 0) < 0.4 and self.preload_distance > 1:
            # 缓存命中率低时减少预加载距离
            self.preload_distance -= 0.5

        # 确保预加载距离在合理范围内
        self.preload_distance = int(max(1, min(self.preload_distance, 4)))


    def _generate_chunk(self, chunk_pos):
        """生成区块 - 实现增量生成逻辑，分步减轻主线程负担"""
        # 初始化生成进度
        if chunk_pos not in self.chunk_generation_progress:
            self.chunk_generation_progress[chunk_pos] = 0
            # 初始化区块数据
            self.chunk_generation_data[chunk_pos] = {
                'blocks': [],
                'heightmap': [],
                'structures': [],
                'terrain_generated': False,
                'structures_generated': False,
                'mesh_generated': False
            }

        progress = self.chunk_generation_progress[chunk_pos]
        chunk_data = self.chunk_generation_data[chunk_pos]
        steps_processed = 0

        # 分步骤生成区块
        try:
            # 步骤1: 生成地形高度图 (0-20步)
            if progress < 20 and not chunk_data['terrain_generated']:
                # 每帧只生成部分高度图数据
                for i in range(progress, min(progress + 5, 20)):
                    x = i % 16
                    z = i // 16
                    # 生成高度数据
                    height = self._generate_height(x, z, chunk_pos)
                    chunk_data['heightmap'].append((x, z, height))
                    steps_processed += 1
                self.chunk_generation_progress[chunk_pos] += steps_processed
                if self.chunk_generation_progress[chunk_pos] >= 20:
                    chunk_data['terrain_generated'] = True
                    progress = 20

            # 步骤2: 生成区块结构 (20-40步)
            if progress >= 20 and progress < 40 and not chunk_data['structures_generated']:
                # 每帧只生成部分结构
                for i in range(progress - 20, min(progress - 20 + 5, 20)):
                    # 生成结构数据
                    structure = self._generate_structure(i, chunk_pos)
                    if structure:
                        chunk_data['structures'].append(structure)
                    steps_processed += 1
                self.chunk_generation_progress[chunk_pos] += steps_processed
                if self.chunk_generation_progress[chunk_pos] >= 40:
                    chunk_data['structures_generated'] = True
                    progress = 40

            # 步骤3: 生成方块数据 (40-60步)
            if progress >= 40 and progress < 60 and not chunk_data['mesh_generated']:
                # 每帧只生成部分方块
                for i in range(progress - 40, min(progress - 40 + 10, 20)):
                    # 生成方块数据
                    blocks = self._generate_blocks(i, chunk_data['heightmap'], chunk_data['structures'])
                    chunk_data['blocks'].extend(blocks)
                    steps_processed += 1
                self.chunk_generation_progress[chunk_pos] += steps_processed
                if self.chunk_generation_progress[chunk_pos] >= 60:
                    chunk_data['mesh_generated'] = True

            # 如果所有生成步骤完成
            if chunk_data['terrain_generated'] and chunk_data['structures_generated'] and chunk_data['mesh_generated']:
                # 创建最终区块对象
                final_chunk = self._create_chunk_object(chunk_pos, chunk_data)
                # 将完成的区块添加到已加载列表
                self.loaded_chunks[chunk_pos] = final_chunk
                self.stats['chunks_loaded_total'] += 1
                # 清理临时数据
                del self.chunk_generation_progress[chunk_pos]
                del self.chunk_generation_data[chunk_pos]
                return final_chunk
            else:
                # 未完成，返回None表示需要继续生成
                return steps_processed
        except Exception as e:
            logging.error(f"区块生成错误: {e}")
            # 清理错误的生成数据
            if chunk_pos in self.chunk_generation_progress:
                del self.chunk_generation_progress[chunk_pos]
            if chunk_pos in self.chunk_generation_data:
                del self.chunk_generation_data[chunk_pos]
            return None

    def _generate_height(self, x, z, chunk_pos):
        """生成地形高度数据"""
        # 简化的高度生成逻辑
        return int(np.sin(x/5) * np.cos(z/5) * 5 + 10)

    def _generate_structure(self, index, chunk_pos):
        """生成结构数据"""
        # 简化的结构生成逻辑
        if index % 7 == 0:  # 随机生成一些结构
            return {
                'type': 'tree',
                'position': (index % 16, 10, index // 16)
            }
        return None

    def _generate_blocks(self, index, heightmap, structures):
        """生成方块数据"""
        # 简化的方块生成逻辑
        blocks = []
        x = index % 16
        z = index // 16
        # 查找该位置的高度
        height = next((h for (hx, hz, h) in heightmap if hx == x and hz == z), 10)
        # 添加方块
        for y in range(height):
            blocks.append((x, y, z, 'grass_block' if y == height-1 else 'dirt_block'))
        return blocks

    def _create_chunk_object(self, chunk_pos, chunk_data):
        """创建最终的区块对象"""
        from loading_system import Chunk  # 从加载系统导入区块类
        # 创建实际的区块对象
        chunk = Chunk(position=chunk_pos)
        chunk.blocks = chunk_data['blocks']
        chunk.structures = chunk_data['structures']
        chunk.generate_mesh()  # 生成区块网格
        return chunk
        # 首先检查缓存
        cached_data = block_cache.get_block_data(chunk_pos)
        if cached_data is not None:
            return cached_data

        # 检查是否已有生成中的数据
        if chunk_pos not in self.chunk_generation_data:
            # 初始化生成数据
            self.chunk_generation_data[chunk_pos] = {
                'stage': 0,  # 0:初始化, 1:高度图, 2:主要方块, 3:细节, 4:完成
                'progress': 0,
                'data': None
            }
            logging.debug(f"开始生成区块 {chunk_pos}")

        gen_data = self.chunk_generation_data[chunk_pos]
        result = None

        try:
            # 根据当前阶段执行部分生成工作
            if gen_data['stage'] == 0:
                # 阶段0: 初始化区块数据
                gen_data['data'] = self._initialize_chunk_data(chunk_pos)
                gen_data['stage'] = 1
                gen_data['progress'] = 20
            elif gen_data['stage'] == 1:
                # 阶段1: 生成高度图
                self._generate_heightmap(gen_data['data'], chunk_pos)
                gen_data['stage'] = 2
                gen_data['progress'] = 40
            elif gen_data['stage'] == 2:
                # 阶段2: 放置主要方块
                self._place_main_blocks(gen_data['data'], chunk_pos)
                gen_data['stage'] = 3
                gen_data['progress'] = 60
            elif gen_data['stage'] == 3:
                # 阶段3: 添加细节和装饰
                self._add_details_and_decorations(gen_data['data'], chunk_pos)
                gen_data['stage'] = 4
                gen_data['progress'] = 80
            elif gen_data['stage'] == 4:
                # 阶段4: 完成生成并缓存
                result = self._finalize_chunk(gen_data['data'], chunk_pos)
                block_cache.cache_block_data(chunk_pos, result)
                del self.chunk_generation_data[chunk_pos]
                gen_data['progress'] = 100
                logging.debug(f"区块 {chunk_pos} 生成完成")

            # 记录生成进度
            self.chunk_generation_progress[chunk_pos] = gen_data['progress']
            return result

        except Exception as e:
            logging.error(f"区块生成错误 {chunk_pos}: {e}")
            if chunk_pos in self.chunk_generation_data:
                del self.chunk_generation_data[chunk_pos]
            return None

    def _initialize_chunk_data(self, chunk_pos):
        """初始化区块数据结构"""
        return {
            'position': chunk_pos,
            'blocks': np.zeros((16, 256, 16), dtype=np.uint8),
            'heightmap': np.zeros((16, 16), dtype=np.uint8),
            'biomes': np.zeros((16, 16), dtype=np.uint8),
            'modified': False
        }

    def _generate_heightmap(self, chunk_data, chunk_pos):
        """生成地形高度图"""
        # 使用简单的噪声算法生成高度图
        x, z = chunk_pos
        for i in range(16):
            for j in range(16):
                world_x = x * 16 + i
                world_z = z * 16 + j
                # 简化的高度计算，减少计算复杂度
                height = int((np.sin(world_x * 0.1) * np.cos(world_z * 0.1) + 1) * 10) + 60
                chunk_data['heightmap'][i, j] = height

    def _place_main_blocks(self, chunk_data, chunk_pos):
        """放置主要方块类型"""
        # 只放置基础方块，减少计算量
        for x in range(16):
            for z in range(16):
                height = chunk_data['heightmap'][x, z]
                # 只放置地面和石头层
                if height > 65:
                    chunk_data['blocks'][x, height, z] = 2  # 草方块
                    for y in range(height-1, height-4, -1):
                        if y > 0:
                            chunk_data['blocks'][x, y, z] = 3  # 泥土
                # 简单的石头层
                for y in range(height-4, max(0, height-10), -1):
                    chunk_data['blocks'][x, y, z] = 1  # 石头

    def _add_details_and_decorations(self, chunk_data, chunk_pos):
        """添加细节和装饰性元素"""
        # 简化细节生成，只添加少量元素
        x, z = chunk_pos
        for i in range(16):
            for j in range(16):
                height = chunk_data['heightmap'][i, j]
                # 偶尔添加树木
                if height > 65 and (x*16+i + z*16+j) % 30 == 0:
                    self._place_tree(chunk_data, i, height+1, j)

    def _place_tree(self, chunk_data, x, y, z):
        """放置简单树木"""
        # 限制树木高度，减少计算
        if y + 4 >= 256:  # 防止超出世界高度
            return
        # 树干
        for dy in range(4):
            chunk_data['blocks'][x, y+dy, z] = 17  # 木头
        # 树叶
        for dy in range(2):
            for dx in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if 0 <= x+dx < 16 and 0 <= z+dz < 16:
                        chunk_data['blocks'][x+dx, y+3+dy, z+dz] = 18  # 树叶

    def _finalize_chunk(self, chunk_data, chunk_pos):
        """完成区块生成并准备返回数据"""
        # 简单的区块数据打包
        return {
            'blocks': chunk_data['blocks'],
            'heightmap': chunk_data['heightmap'],
            'biomes': chunk_data['biomes'],
            'position': chunk_pos
        }

    def _get_chunk_distance(self, chunk_pos1, chunk_pos2):
        """计算两个区块之间的距离"""
        return abs(chunk_pos1[0] - chunk_pos2[0]) + abs(chunk_pos1[1] - chunk_pos2[1])

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

# 辅助函数 - 集成到主循环
def integrate_with_game_loop(player, delta_time):
    """将区块加载优化器集成到游戏主循环"""
    # 获取玩家位置和朝向
    player_position = player.position
    player_direction = player.forward
    
    # 更新区块加载优化器
    chunk_loading_optimizer.update(player_position, player_direction, delta_time)
    
    # 增量区块生成 - 将单个区块生成任务分解为多个步骤
    chunk_loading_optimizer.chunk_generation_steps = defaultdict(int)  # 记录每个区块的生成步骤
    chunk_loading_optimizer.max_steps_per_frame = 5  # 每帧最多处理5个生成步骤
    chunk_loading_optimizer._process_incremental_generation()

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