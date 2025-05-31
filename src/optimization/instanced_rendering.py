# 实例化渲染优化模块
from ursina import *
import time
from collections import defaultdict

class InstancedRenderer:
    """实例化渲染管理器，使用instanced_cube模型进行高效渲染"""
    
    def __init__(self):
        self.block_groups = defaultdict(list)  # 按方块ID分组
        self.combined_entities = {}  # 存储合并后的实体
        self.dirty_groups = set()  # 需要更新的方块组
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒），从1.0降低到0.05以提高响应性
    
    def add_block(self, block):
        """添加方块到渲染组"""
        block_id = block.id
        self.block_groups[block_id].append(block)
        self.dirty_groups.add(block_id)
    
    def remove_block(self, block):
        """从渲染组中移除方块"""
        block_id = block.id
        if block in self.block_groups[block_id]:
            self.block_groups[block_id].remove(block)
            self.dirty_groups.add(block_id)
    
    def update(self, force=False):
        """更新需要重新渲染的方块组"""
        current_time = time.time()
        
        # 降低更新频率
        if not force and current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 限制每次更新处理的脏组数量，避免卡顿
        dirty_groups_to_process = list(self.dirty_groups)[:10]  # 每次最多处理10个组
        
        # 更新所有标记为脏的方块组
        for block_id in dirty_groups_to_process:
            blocks = self.block_groups[block_id]
            
            # 如果没有方块，销毁合并实体并继续
            if not blocks:
                if block_id in self.combined_entities:
                    destroy(self.combined_entities[block_id])
                    del self.combined_entities[block_id]
                self.dirty_groups.remove(block_id)
                continue
            
            # 销毁旧的合并实体
            if block_id in self.combined_entities:
                destroy(self.combined_entities[block_id])
            
            # 创建新的实例化实体并合并方块
            try:
                # 使用第一个方块的纹理
                texture = blocks[0].texture
                
                # 创建实例化实体
                combined = Entity(model='instanced_cube')
                
                # 设置纹理
                if texture:
                    combined.texture = texture
                
                # 合并方块，保留原始实体
                combined.combine(blocks, auto_destroy=False, keep_origin=True)
                
                # 存储合并后的实体
                self.combined_entities[block_id] = combined
                
                # 隐藏原始方块的视觉效果，但保留碰撞体积
                for block in blocks:
                    # 保存原始颜色和纹理
                    if not hasattr(block, '_original_color'):
                        block._original_color = block.color
                        block._original_texture = block.texture
                    
                    # 隐藏视觉效果但保留碰撞
                    block.color = color.clear
                    block.texture = None
            except Exception as e:
                print(f"合并方块错误: {e}")
            
            # 从脏列表中移除
            self.dirty_groups.remove(block_id)
    
    def restore_blocks(self):
        """恢复所有方块的原始视觉效果"""
        for block_id, blocks in self.block_groups.items():
            for block in blocks:
                if hasattr(block, '_original_color'):
                    block.color = block._original_color
                if hasattr(block, '_original_texture'):
                    block.texture = block._original_texture
        
        # 销毁所有合并实体
        for entity in self.combined_entities.values():
            destroy(entity)
        
        self.combined_entities.clear()
        self.dirty_groups.clear()

# 网格合并管理器
class MeshCombiner:
    """网格合并管理器，用于合并相同类型的方块网格"""
    
    def __init__(self, chunk_size=8):
        self.chunk_size = chunk_size
        self.combined_chunks = {}  # 存储已合并的区块
        self.dirty_chunks = set()  # 需要重新合并的区块
    
    def _get_chunk_coords(self, position):
        """获取方块所在的区块坐标"""
        return (
            int(position.x // self.chunk_size),
            int(position.y // self.chunk_size),
            int(position.z // self.chunk_size)
        )
    
    def add_block(self, block):
        """添加方块并标记其所在区块为脏"""
        chunk_coords = self._get_chunk_coords(block.position)
        self.dirty_chunks.add(chunk_coords)
    
    def remove_block(self, block):
        """移除方块并标记其所在区块为脏"""
        chunk_coords = self._get_chunk_coords(block.position)
        self.dirty_chunks.add(chunk_coords)
    
    def update(self, blocks_by_chunk):
        """更新所有脏区块的合并网格"""
        # 限制每次更新处理的脏区块数量
        dirty_chunks_to_process = list(self.dirty_chunks)[:5]  # 每次最多处理5个区块
        
        for chunk_coords in dirty_chunks_to_process:
            # 获取该区块的所有方块
            if chunk_coords not in blocks_by_chunk or not blocks_by_chunk[chunk_coords]:
                # 如果区块为空，销毁合并实体
                if chunk_coords in self.combined_chunks:
                    destroy(self.combined_chunks[chunk_coords])
                    del self.combined_chunks[chunk_coords]
                self.dirty_chunks.remove(chunk_coords)
                continue
            
            # 获取区块中的方块
            blocks = blocks_by_chunk[chunk_coords]
            
            # 销毁旧的合并实体
            if chunk_coords in self.combined_chunks:
                destroy(self.combined_chunks[chunk_coords])
            
            # 创建新的合并实体
            combined = Entity(model='cube')
            combined.combine(blocks, auto_destroy=False, keep_origin=True)
            
            # 存储合并后的实体
            self.combined_chunks[chunk_coords] = combined
            
            # 从脏列表中移除
            self.dirty_chunks.remove(chunk_coords)