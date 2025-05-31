# 区块状态管理器
# 用于管理区块的状态和玩家修改

import logging
from collections import defaultdict

class ChunkStateManager:
    """区块状态管理器 - 负责管理和持久化区块状态"""
    
    def __init__(self):
        # 区块状态字典，使用defaultdict避免KeyError
        self.chunk_states = defaultdict(dict)
        # 区块修改记录，记录玩家对区块的修改
        self.chunk_modifications = defaultdict(list)
        
    def save_chunk_state(self, chunk_pos, chunk):
        """保存区块状态
        
        Args:
            chunk_pos: 区块坐标
            chunk: 区块对象
        """
        try:
            # 保存区块中所有方块的状态
            block_states = []
            for block in chunk.blocks:
                if block:
                    block_state = {
                        'position': (block.position.x, block.position.y, block.position.z),
                        'id': block.id,
                        'modified': True  # 标记为玩家修改
                    }
                    block_states.append(block_state)
            
            # 保存到状态字典
            self.chunk_states[chunk_pos] = {
                'blocks': block_states,
                'last_modified': time.time()
            }
            
            logging.debug(f"已保存区块 {chunk_pos} 的状态")
            
        except Exception as e:
            logging.error(f"保存区块 {chunk_pos} 状态时出错: {e}")
    
    def load_chunk_state(self, chunk_pos):
        """加载区块状态
        
        Args:
            chunk_pos: 区块坐标
            
        Returns:
            dict: 区块状态数据，如果不存在则返回None
        """
        return self.chunk_states.get(chunk_pos)
    
    def record_block_modification(self, chunk_pos, block_pos, block_id, action='add'):
        """记录方块修改
        
        Args:
            chunk_pos: 区块坐标
            block_pos: 方块坐标
            block_id: 方块ID
            action: 操作类型('add'或'remove')
        """
        modification = {
            'position': block_pos,
            'id': block_id,
            'action': action,
            'time': time.time()
        }
        self.chunk_modifications[chunk_pos].append(modification)
    
    def apply_modifications(self, chunk_pos, chunk):
        """应用区块修改
        
        Args:
            chunk_pos: 区块坐标
            chunk: 区块对象
        """
        if chunk_pos in self.chunk_modifications:
            for mod in self.chunk_modifications[chunk_pos]:
                try:
                    if mod['action'] == 'add':
                        # 添加方块
                        new_block = Block(position=mod['position'], id=mod['id'], use_mesh_splitting=True)
                        chunk.blocks.append(new_block)
                    elif mod['action'] == 'remove':
                        # 移除方块
                        for block in chunk.blocks:
                            if block.position == mod['position']:
                                chunk.blocks.remove(block)
                                break
                except Exception as e:
                    logging.error(f"应用区块修改时出错: {e}")
    
    def clear_chunk_state(self, chunk_pos):
        """清除区块状态
        
        Args:
            chunk_pos: 区块坐标
        """
        if chunk_pos in self.chunk_states:
            del self.chunk_states[chunk_pos]
        if chunk_pos in self.chunk_modifications:
            del self.chunk_modifications[chunk_pos]
    
    def get_modified_chunks(self):
        """获取所有被修改的区块坐标
        
        Returns:
            list: 被修改的区块坐标列表
        """
        return list(self.chunk_modifications.keys())

# 创建全局实例
chunk_state_manager = ChunkStateManager()