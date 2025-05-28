# 区块优化模块
from ursina import *
import time
from math import floor
from collections import deque

# 区块管理系统
class ChunkManager:
    def __init__(self, chunk_size=8, render_distance=2, check_interval=60):
        self.chunks = {}  # 用字典存储区块, key 是区块的坐标 (x, y, z)
        self.chunk_size = chunk_size
        self.render_distance = render_distance
        self.check_interval = check_interval
        self.frame_count = 0
        self.last_check_time = 0
        self.check_interval_seconds = 1.0  # 每秒检查一次
        
    def _get_chunk_coords(self, position):
        """根据世界坐标获取区块坐标"""
        return (
            floor(position.x / self.chunk_size),
            floor(position.y / self.chunk_size),
            floor(position.z / self.chunk_size)
        )
    
    def add_block_to_chunk(self, block):
        """将方块添加到对应的区块"""
        chunk_coords = self._get_chunk_coords(block.position)
        if chunk_coords not in self.chunks:
            self.chunks[chunk_coords] = []
        self.chunks[chunk_coords].append(block)
        
    def remove_block_from_chunk(self, block):
        """从区块中移除方块"""
        chunk_coords = self._get_chunk_coords(block.position)
        if chunk_coords in self.chunks and block in self.chunks[chunk_coords]:
            self.chunks[chunk_coords].remove(block)
            # 如果区块为空，可以选择性地删除该区块
            if not self.chunks[chunk_coords]:
                del self.chunks[chunk_coords]
    
    def update_chunks_visibility(self, player_position):
        """更新区块可见性，根据与玩家的距离"""
        current_time = time.time()
        
        # 降低检查频率，不必每帧都检查
        if current_time - self.last_check_time < self.check_interval_seconds:
            return
            
        self.last_check_time = current_time
        player_chunk = self._get_chunk_coords(player_position)
        
        for chunk_coords, blocks in self.chunks.items():
            # 计算区块中心与玩家区块的距离
            dx = abs(chunk_coords[0] - player_chunk[0])
            dy = abs(chunk_coords[1] - player_chunk[1])
            dz = abs(chunk_coords[2] - player_chunk[2])
            
            # 使用曼哈顿距离作为简单的距离度量
            distance = dx + dy + dz
            
            # 根据距离决定区块是否可见
            visible = distance <= self.render_distance
            
            # 更新区块中所有方块的可见性
            for block in blocks:
                if visible:
                    block.enable()
                else:
                    block.disable()

# 实例化渲染管理器
class InstancedRenderManager:
    def __init__(self):
        self.instanced_entities = {}
        self.block_types = {}  # 存储不同类型的方块
        
    def add_block(self, block):
        """添加方块到实例化渲染管理器"""
        block_id = block.id
        if block_id not in self.block_types:
            # 为每种方块类型创建一个实例化实体
            self.block_types[block_id] = Entity(model='instanced_cube', texture=block.texture)
            self.instanced_entities[block_id] = []
        
        # 添加方块到对应类型的列表
        self.instanced_entities[block_id].append(block)
        
    def remove_block(self, block):
        """从实例化渲染管理器中移除方块"""
        block_id = block.id
        if block_id in self.instanced_entities and block in self.instanced_entities[block_id]:
            self.instanced_entities[block_id].remove(block)
            
    def update_rendering(self):
        """更新所有实例化渲染"""
        for block_id, blocks in self.instanced_entities.items():
            if blocks and block_id in self.block_types:
                # 清除现有的实例化实体
                destroy(self.block_types[block_id])
                
                # 创建新的实例化实体
                self.block_types[block_id] = Entity(model='instanced_cube', texture=blocks[0].texture)
                
                # 合并所有同类型的方块
                self.block_types[block_id].combine(blocks, auto_destroy=False, keep_origin=True)

# 网格合并管理器
class MeshCombineManager:
    def __init__(self, chunk_size=8):
        self.chunk_size = chunk_size
        self.combined_chunks = {}  # 存储已合并的区块
        self.dirty_chunks = set()  # 存储需要重新合并的区块
        
    def mark_chunk_dirty(self, chunk_coords):
        """标记区块为脏，需要重新合并"""
        self.dirty_chunks.add(chunk_coords)
        
    def combine_chunk(self, chunk_coords, blocks):
        """合并一个区块内的所有方块"""
        if not blocks:
            return
            
        # 如果该区块已经有合并的网格，先销毁
        if chunk_coords in self.combined_chunks:
            destroy(self.combined_chunks[chunk_coords])
            
        # 创建新的合并网格
        combined_entity = Entity(model='cube')
        combined_entity.combine(blocks, auto_destroy=False, keep_origin=True)
        
        # 存储合并后的实体
        self.combined_chunks[chunk_coords] = combined_entity
        
    def update(self, chunk_manager):
        """更新所有脏区块的合并网格"""
        for chunk_coords in list(self.dirty_chunks):
            if chunk_coords in chunk_manager.chunks:
                self.combine_chunk(chunk_coords, chunk_manager.chunks[chunk_coords])
            self.dirty_chunks.remove(chunk_coords)

# 距离检查优化器
class DistanceCheckOptimizer:
    def __init__(self, check_interval=30):
        self.check_interval = check_interval  # 检查间隔（帧数）
        self.frame_count = 0
        
    def update(self, player, blocks, max_distance=10):
        """更新方块可见性，基于与玩家的距离"""
        self.frame_count += 1
        
        # 降低检查频率，不必每帧都检查
        if self.frame_count >= self.check_interval:
            self.frame_count = 0
            
            for block in blocks:
                # 计算与玩家的距离
                dist = distance(player.position, block.position)
                
                # 根据距离设置方块可见性
                if dist > max_distance:
                    block.disable()
                else:
                    block.enable()