# LOD系统模块
# 用于根据距离动态调整方块渲染细节

from ursina import *
import time

# LOD级别定义 - 极度降低距离阈值以提高性能
LOD_LEVELS = [
    {'distance': 2, 'name': '高细节', 'collision': True, 'update_interval': 0.5},  # 近距离 - 高细节，距离从3降低到2
    {'distance': 5, 'name': '中细节', 'collision': True, 'update_interval': 1.0},   # 中距离 - 中细节，距离从6降低到5
    {'distance': 10, 'name': '低细节', 'collision': False, 'update_interval': 2.0}   # 远距离 - 低细节，距离从12降低到10
]

class LODManager:
    """LOD管理器，根据距离动态调整方块渲染细节，针对16区块渲染距离优化"""
    
    def __init__(self):
        self.enabled = True
        self.update_interval = 0.2  # 更新间隔（秒）从0.5减小到0.2
        self.last_update_time = 0
        
        # LOD级别配置 - 针对16区块渲染距离优化
        self.lod_levels = {
            0: {'distance': 0, 'detail': 1.0},     # 最高细节 - 近距离
            1: {'distance': 32, 'detail': 0.8},   # 高细节 - 2区块
            2: {'distance': 64, 'detail': 0.6},   # 中高细节 - 4区块
            3: {'distance': 128, 'detail': 0.4},  # 中细节 - 8区块
            4: {'distance': 192, 'detail': 0.2},  # 低细节 - 12区块
            5: {'distance': 256, 'detail': 0.1}   # 最低细节 - 16区块
        }
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
        """根据距离获取LOD级别，针对16区块渲染距离优化"""
        if not self.enabled:
            return 0  # 如果禁用LOD，始终返回最高细节
        
        # 根据距离确定LOD级别
        if distance < 32:  # 2区块
            return 0  # 最高细节
        elif distance < 64:  # 4区块
            return 1  # 高细节
        elif distance < 128:  # 8区块
            return 2  # 中高细节
        elif distance < 192:  # 12区块
            return 3  # 中细节
        elif distance < 256:  # 16区块
            return 4  # 低细节
        else:
            return 5  # 最低细节
    
    def apply_lod(self, block, distance_to_player):
        """应用LOD到方块，针对16区块渲染距离优化"""
        if not self.enabled or not hasattr(block, 'lod_level'):
            return
        
        # 获取当前LOD级别
        current_lod = self.get_lod_level(distance_to_player)
        
        # 如果LOD级别没有变化，不需要更新
        if block.lod_level == current_lod:
            return False
        
        # 更新LOD级别
        block.lod_level = current_lod
        
        # 更新统计信息
        if current_lod == 0:
            self.stats['high_detail'] += 1
        elif current_lod <= 2:
            self.stats['medium_detail'] += 1
        else:
            self.stats['low_detail'] += 1
        
        # 应用LOD效果
        if current_lod == 0:  # 最高细节 - 近距离
            # 启用所有细节
            block.scale = Vec3(1, 1, 1)
            block.alpha = 1.0
            block.model = 'cube'  # 高细节模型
            block.collision = True
            block.update_interval = 0.5
        elif current_lod == 1:  # 高细节 - 2区块
            # 禁用一些高消耗细节
            block.scale = Vec3(0.95, 0.95, 0.95)
            block.alpha = 0.9
            block.model = 'cube_low'  # 中等细节模型
            block.collision = True
            block.update_interval = 1.0
        elif current_lod == 2:  # 中高细节 - 4区块
            # 适度减少细节
            block.scale = Vec3(0.9, 0.9, 0.9)
            block.alpha = 0.8
            block.model = 'quad'  # 远距离使用平面模型
            block.collision = False
            block.update_interval = 2.0
        elif current_lod == 3:  # 中细节 - 8区块
            # 进一步减少细节
            block.scale = Vec3(0.85, 0.85, 0.85)
            block.alpha = 0.7
            block.model = 'quad'
            block.collision = False
            block.update_interval = 3.0
        elif current_lod == 4:  # 低细节 - 12区块
            # 大幅减少细节
            block.scale = Vec3(0.8, 0.8, 0.8)
            block.alpha = 0.6
            block.model = 'quad'
            block.collision = False
            block.update_interval = 4.0
        else:  # 最低细节 - 16区块
            # 最小化细节
            block.scale = Vec3(0.7, 0.7, 0.7)
            block.alpha = 0.5
            block.model = 'quad'
            block.collision = False
            block.update_interval = 5.0
        
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