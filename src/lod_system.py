# LOD系统模块
# 用于根据距离动态调整方块渲染细节

from ursina import *
import time

# LOD级别定义 - 极度降低距离阈值以提高性能
LOD_LEVELS = [
    {'distance': 3, 'name': '高细节', 'collision': True, 'update_interval': 0.5},  # 近距离 - 高细节
    {'distance': 6, 'name': '中细节', 'collision': True, 'update_interval': 1.0},   # 中距离 - 中细节
    {'distance': 12, 'name': '低细节', 'collision': False, 'update_interval': 2.0}   # 远距离 - 低细节
]

class LODManager:
    """LOD管理器，根据距离动态调整方块渲染细节"""
    
    def __init__(self):
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.2  # 减少全局更新间隔以减少CPU负担
        self.stats = {
            'high_detail': 0,
            'medium_detail': 0,
            'low_detail': 0,
            'total_blocks': 0
        }
    
    def update(self):
        """更新LOD系统"""
        if not self.enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        # 动态调整LOD距离阈值
        # 根据当前帧率动态调整LOD距离阈值
        # 如果帧率低，减小LOD距离阈值，提高性能
        # 如果帧率高，增加LOD距离阈值，提高视觉质量
        from ursina import application
        current_fps = getattr(application, 'fps', 30)
        
        # 帧率阈值 - 更激进的阈值设置
        LOW_FPS = 20  # 降低低帧率阈值，更早开始优化
        TARGET_FPS = 30  # 目标帧率
        HIGH_FPS = 40  # 降低高帧率阈值，确保有足够余量再提高视觉质量
        
        # 根据帧率调整LOD距离 - 更激进的调整策略
        if current_fps < LOW_FPS:
            # 帧率很低，极大幅度减小LOD距离阈值
            adjustment_factor = max(0.3, (current_fps / LOW_FPS))  # 更激进的调整因子
            LOD_LEVELS[0]['distance'] = max(2, int(LOD_LEVELS[0]['distance'] * adjustment_factor))
            LOD_LEVELS[1]['distance'] = max(4, int(LOD_LEVELS[1]['distance'] * adjustment_factor))
            # 增加更新间隔，减少CPU负担
            self.update_interval = min(0.5, self.update_interval * 1.1)
        elif current_fps < TARGET_FPS:
            # 帧率低于目标，适度减小LOD距离阈值
            adjustment_factor = max(0.5, (current_fps / TARGET_FPS))
            LOD_LEVELS[0]['distance'] = max(3, int(LOD_LEVELS[0]['distance'] * adjustment_factor))
            LOD_LEVELS[1]['distance'] = max(6, int(LOD_LEVELS[1]['distance'] * adjustment_factor))
            # 适度增加更新间隔
            self.update_interval = min(0.3, self.update_interval * 1.05)
        elif current_fps > HIGH_FPS:
            # 帧率高，适度增加LOD距离阈值
            LOD_LEVELS[0]['distance'] = min(6, LOD_LEVELS[0]['distance'] + 1)
            LOD_LEVELS[1]['distance'] = min(12, LOD_LEVELS[1]['distance'] + 1)
            # 减少更新间隔，提高响应性
            self.update_interval = max(0.15, self.update_interval * 0.95)
    
    def get_lod_level(self, distance):
        """根据距离获取LOD级别"""
        if not self.enabled:
            return 0  # 如果LOD系统禁用，则返回最高细节级别
        
        for i, level in enumerate(LOD_LEVELS):
            if distance <= level['distance']:
                return i
        
        # 如果距离超过所有级别，返回最低细节级别
        return len(LOD_LEVELS) - 1
    
    def apply_lod(self, block, distance_to_player):
        """应用LOD设置到方块"""
        if not self.enabled or not hasattr(block, 'lod_level'):
            return
        
        # 获取当前LOD级别
        current_lod = self.get_lod_level(distance_to_player)
        
        # 如果LOD级别没有变化，不需要更新
        if block.lod_level == current_lod:
            return False
        
        # 更新LOD级别
        block.lod_level = current_lod
        lod_config = LOD_LEVELS[current_lod]
        
        # 应用LOD设置
        # 1. 碰撞检测
        block.collision = lod_config['collision']
        
        # 2. 更新间隔
        block.update_interval = lod_config['update_interval']
        
        # 3. 纹理质量 (可以在这里实现纹理降采样)
        # 这里简化处理，实际项目中可以使用不同分辨率的纹理
        
        return True  # 返回True表示LOD级别已更新
    
    def update_stats(self, blocks):
        """更新LOD统计信息"""
        if not self.enabled:
            return
        
        # 重置统计信息
        self.stats['high_detail'] = 0
        self.stats['medium_detail'] = 0
        self.stats['low_detail'] = 0
        self.stats['total_blocks'] = len(blocks)
        
        # 统计各LOD级别的方块数量
        for block in blocks:
            if hasattr(block, 'lod_level'):
                if block.lod_level == 0:
                    self.stats['high_detail'] += 1
                elif block.lod_level == 1:
                    self.stats['medium_detail'] += 1
                else:
                    self.stats['low_detail'] += 1

# 全局LOD管理器实例
lod_manager = LODManager()

# 扩展Block类，添加LOD支持
def initialize_block_lod(block):
    """初始化方块的LOD属性"""
    if not hasattr(block, 'lod_level'):
        block.lod_level = 0  # 默认使用最高细节级别

# 批量处理方块LOD
def process_blocks_lod(blocks, player_position):
    """批量处理方块的LOD级别"""
    if not lod_manager.enabled:
        return blocks
    
    current_time = time.time()
    if current_time - lod_manager.last_update_time < lod_manager.update_interval:
        return blocks  # 如果未到更新时间，直接返回
    
    lod_manager.last_update_time = current_time
    
    # 处理每个方块的LOD
    for block in blocks:
        if not hasattr(block, 'distance_to_player'):
            # 计算方块到玩家的距离
            block.distance_to_player = (block.position - player_position).length()
        
        # 初始化LOD属性
        initialize_block_lod(block)
        
        # 应用LOD设置
        lod_manager.apply_lod(block, block.distance_to_player)
    
    # 更新统计信息
    lod_manager.update_stats(blocks)
    
    return blocks