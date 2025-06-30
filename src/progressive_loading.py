# 渐进式资源加载系统
# 解决游戏开始时帧率高但无法持续的问题

import time
import threading
import logging
from queue import PriorityQueue, Queue, Empty
from collections import defaultdict, deque
import numpy as np
from ursina import Vec3, invoke
import gc

# 导入相关系统
from loading_system import chunk_loader, get_chunk_position
from block_cache import block_cache
from performance_config import perf_config

class ProgressiveLoadingSystem:
    """渐进式资源加载系统 - 实现动态资源加载和卸载，平衡初始帧率和持续性能"""
    
    def __init__(self):
        # 基础参数
        self.enabled = True
        self.last_update_time = 0
        self.startup_phase = True  # 标记游戏启动阶段
        self.startup_timer = 0     # 启动阶段计时器
        self.startup_duration = 10.0  # 启动阶段持续时间（秒）
        
        # 加载队列和状态
        self.immediate_queue = PriorityQueue()  # 立即加载队列（高优先级）
        self.background_queue = PriorityQueue()  # 后台加载队列（低优先级）
        self.loading_chunks = set()  # 正在加载的区块
        
        # 性能监控
        self.fps_history = deque(maxlen=30)  # 记录最近30帧的FPS
        self.frame_time_history = deque(maxlen=30)  # 记录最近30帧的帧时间
        self.target_fps = 60  # 目标FPS
        
        # 加载控制参数
        self.initial_load_limit = 1  # 启动阶段每帧加载区块数
        self.normal_load_limit = 2   # 正常阶段每帧加载区块数
        self.background_load_interval = 0.5  # 后台加载间隔（秒）
        self.last_background_load_time = 0
        
        # 渐进式加载参数
        self.load_stages = [
            {"distance": 1, "priority": 0.1},  # 第一阶段：玩家周围1个区块
            {"distance": 2, "priority": 0.5},  # 第二阶段：玩家周围2个区块
            {"distance": 3, "priority": 1.0},  # 第三阶段：玩家周围3个区块
            {"distance": 4, "priority": 2.0},  # 第四阶段：玩家周围4个区块
            {"distance": 5, "priority": 5.0}   # 第五阶段：玩家周围5个区块
        ]
        self.current_stage = 0  # 当前加载阶段
        
        # 统计信息
        self.stats = {
            "startup_chunks_loaded": 0,
            "background_chunks_loaded": 0,
            "immediate_chunks_loaded": 0,
            "current_stage": 0,
            "startup_progress": 0.0,
            "startup_complete": False,
            "immediate_queue_size": 0,
            "background_queue_size": 0,
            "current_loading_rate": 0,
            "loading_chunks_count": 0
        }
        
        logging.info("渐进式资源加载系统初始化完成")
    
    def update(self, player_position, delta_time=0.016):
        """更新渐进式加载系统"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # 更新启动阶段状态
        if self.startup_phase:
            self.startup_timer += delta_time
            self.stats["startup_progress"] = min(1.0, self.startup_timer / self.startup_duration)
            
            # 启动阶段结束检查
            if self.startup_timer >= self.startup_duration:
                self.startup_phase = False
                logging.info("启动阶段结束，切换到正常加载模式")
        
        # 获取玩家所在区块
        player_chunk = get_chunk_position(player_position)
        
        # 1. 处理立即加载队列（高优先级）
        self._process_immediate_queue(player_chunk)
        
        # 2. 处理后台加载队列（低优先级，间隔执行）
        if current_time - self.last_background_load_time >= self.background_load_interval:
            self._process_background_queue()
            self.last_background_load_time = current_time
        
        # 3. 根据当前阶段更新加载队列
        self._update_loading_stages(player_position)
        
        # 4. 自适应调整加载参数
        self._adapt_loading_parameters()
        
        # 5. 更新统计信息
        self.stats["loading_chunks_count"] = len(self.loading_chunks)
    
    def _process_immediate_queue(self, player_chunk):
        """处理立即加载队列（高优先级）"""
        # 确定本次处理的区块数量
        chunks_to_load = self.initial_load_limit if self.startup_phase else self.normal_load_limit
        
        # 处理队列
        loaded_count = 0
        while not self.immediate_queue.empty() and loaded_count < chunks_to_load:
            try:
                priority, chunk_pos = self.immediate_queue.get_nowait()
                
                # 跳过已在加载的区块
                if chunk_pos in self.loading_chunks:
                    continue
                
                # 标记为正在加载
                self.loading_chunks.add(chunk_pos)
                
                # 提交加载任务
                self._load_chunk(chunk_pos, priority, immediate=True)
                
                loaded_count += 1
                if self.startup_phase:
                    self.stats["startup_chunks_loaded"] += 1
                else:
                    self.stats["immediate_chunks_loaded"] += 1
            except Empty:
                break
    
    def _process_background_queue(self):
        """处理后台加载队列（低优先级）"""
        # 后台每次只加载一个区块
        if not self.background_queue.empty():
            try:
                priority, chunk_pos = self.background_queue.get_nowait()
                
                # 跳过已在加载的区块
                if chunk_pos in self.loading_chunks:
                    return
                
                # 标记为正在加载
                self.loading_chunks.add(chunk_pos)
                
                # 提交加载任务
                self._load_chunk(chunk_pos, priority, immediate=False)
                
                self.stats["background_chunks_loaded"] += 1
            except Empty:
                pass
    
    def _load_chunk(self, chunk_pos, priority, immediate=True):
        """加载区块，可以是立即加载或延迟加载"""
        # 使用invoke延迟执行，减轻主线程负担
        if immediate:
            # 立即加载，但使用小延迟避免卡顿
            invoke(lambda: self._execute_chunk_load(chunk_pos), delay=0.01)
        else:
            # 后台加载，使用较长延迟
            invoke(lambda: self._execute_chunk_load(chunk_pos), delay=0.1)
    
    def _execute_chunk_load(self, chunk_pos):
        """执行区块加载"""
        try:
            # 调用区块加载系统加载区块
            chunk_loader.load_chunk(chunk_pos)
            
            # 加载完成后从加载集合中移除
            if chunk_pos in self.loading_chunks:
                self.loading_chunks.remove(chunk_pos)
        except Exception as e:
            logging.error(f"加载区块 {chunk_pos} 时出错: {e}")
            # 出错时也要从加载集合中移除
            if chunk_pos in self.loading_chunks:
                self.loading_chunks.remove(chunk_pos)
    
    def _update_loading_stages(self, player_position):
        """根据当前阶段更新加载队列"""
        # 获取玩家所在区块
        player_chunk = get_chunk_position(player_position)
        
        # 确定当前应该加载的阶段
        if self.startup_phase:
            # 启动阶段：根据启动进度确定加载阶段
            progress = self.startup_timer / self.startup_duration
            self.current_stage = min(len(self.load_stages) - 1, int(progress * len(self.load_stages)))
        else:
            # 正常阶段：加载所有阶段
            self.current_stage = len(self.load_stages) - 1
        
        self.stats["current_stage"] = self.current_stage
        
        # 获取当前阶段的加载距离和优先级
        current_distance = self.load_stages[self.current_stage]["distance"]
        current_priority = self.load_stages[self.current_stage]["priority"]
        
        # 生成要加载的区块列表
        chunks_to_load = []
        for dx in range(-current_distance, current_distance + 1):
            for dz in range(-current_distance, current_distance + 1):
                # 计算区块位置
                chunk_pos = (player_chunk[0] + dx, player_chunk[1] + dz)
                
                # 计算到玩家的距离
                distance = max(abs(dx), abs(dz))
                
                # 根据距离确定优先级和队列
                if distance <= 1:
                    # 玩家周围1个区块，最高优先级，立即加载
                    priority = 0.1
                    self.immediate_queue.put((priority, chunk_pos))
                elif distance <= self.current_stage + 1:
                    # 当前阶段内的区块，较高优先级，立即加载
                    priority = current_priority * (distance / (self.current_stage + 1))
                    self.immediate_queue.put((priority, chunk_pos))
                else:
                    # 当前阶段外的区块，低优先级，后台加载
                    priority = current_priority * 2 * (distance / (self.current_stage + 1))
                    self.background_queue.put((priority, chunk_pos))
    
    def _adapt_loading_parameters(self):
        """自适应调整加载参数"""
        # 获取当前FPS
        current_fps = getattr(perf_config, 'current_fps', 60)
        self.fps_history.append(current_fps)
        
        # 计算平均FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 60
        
        # 根据FPS调整加载参数
        if avg_fps < 20:
            # 帧率很低，最小化加载
            self.normal_load_limit = 1
            self.background_load_interval = 1.0
        elif avg_fps < 30:
            # 帧率较低，减少加载
            self.normal_load_limit = 1
            self.background_load_interval = 0.5
        elif avg_fps < 45:
            # 帧率一般，正常加载
            self.normal_load_limit = 2
            self.background_load_interval = 0.3
        else:
            # 帧率良好，增加加载
            self.normal_load_limit = 3
            self.background_load_interval = 0.2
    
    def get_loading_stats(self):
        """获取加载统计信息"""
        # 更新队列大小信息
        self.stats["immediate_queue_size"] = self.immediate_queue.qsize()
        self.stats["background_queue_size"] = self.background_queue.qsize()
        self.stats["loading_chunks_count"] = len(self.loading_chunks)
        self.stats["current_loading_rate"] = self.normal_load_limit if not self.startup_phase else self.initial_load_limit
        self.stats["startup_complete"] = not self.startup_phase
        
        # 更新当前阶段名称
        stage_names = ["近距离", "短距离", "中距离", "远距离", "最远距离"]
        if self.current_stage < len(stage_names):
            self.stats["current_stage"] = stage_names[self.current_stage]
        else:
            self.stats["current_stage"] = "完成"
            
        return self.stats

# 创建全局实例
progressive_loader = ProgressiveLoadingSystem()

# 辅助函数 - 集成到游戏主循环
def integrate_with_game_loop(player, delta_time):
    """将渐进式加载系统集成到游戏主循环"""
    if player and hasattr(player, 'position'):
        progressive_loader.update(player.position, delta_time)

# 辅助函数 - 游戏启动时初始化
def initialize_on_game_start(player_position):
    """游戏启动时初始化并立即加载所有渲染距离内的区块"""
    # 重置状态
    progressive_loader.startup_phase = False  # 直接设置为非启动阶段，跳过渐进式加载
    progressive_loader.startup_timer = progressive_loader.startup_duration  # 设置为已完成启动阶段
    
    # 清空队列
    while not progressive_loader.immediate_queue.empty():
        try:
            progressive_loader.immediate_queue.get_nowait()
        except Empty:
            break
    
    while not progressive_loader.background_queue.empty():
        try:
            progressive_loader.background_queue.get_nowait()
        except Empty:
            break
    
    # 设置为最后阶段，加载所有区块
    progressive_loader.current_stage = len(progressive_loader.load_stages) - 1
    
    # 获取玩家所在区块
    from loading_system import get_chunk_position
    player_chunk = get_chunk_position(player_position)
    
    # 获取当前阶段的加载距离
    current_distance = progressive_loader.load_stages[progressive_loader.current_stage]["distance"]
    
    # 立即加载所有渲染距离内的区块
    chunks_to_load = []
    for dx in range(-current_distance, current_distance + 1):
        for dz in range(-current_distance, current_distance + 1):
            # 计算区块位置
            chunk_pos = (player_chunk[0] + dx, player_chunk[1] + dz)
            # 计算到玩家的距离
            distance = max(abs(dx), abs(dz))
            # 所有区块都设置为最高优先级，立即加载
            priority = 0.1
            progressive_loader.immediate_queue.put((priority, chunk_pos))
    
    # 立即处理队列中的所有区块
    from chunk_loading_optimizer import preload_initial_chunks
    preload_initial_chunks(player_position, distance=current_distance)
    
    # 更新加载阶段
    progressive_loader._update_loading_stages(player_position)
    
    logging.info(f"已立即加载所有渲染距离({current_distance}个区块)内的区块")
    
    # 设置统计信息
    progressive_loader.stats["startup_complete"] = True
    progressive_loader.stats["startup_progress"] = 1.0