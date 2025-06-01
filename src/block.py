# 方块类定义
from ursina import *

class Block(Button):
    use_mesh_splitting_globally = True
    def __init__(self, position=(0, 0, 0), id=0):
        # 如果启用面片渲染，则不创建传统的3D模型
        if Block.use_mesh_splitting_globally:
            # 只创建基础的Button对象用于碰撞检测，不显示模型
            super().__init__(
                parent=scene,
                position=position,
                model=None,  # 不使用模型
                origin_y=0.5,
                texture=None,
                scale=0.5,
                color=color.clear,  # 透明，因为渲染由面片系统处理
                collider='box',
                collision_cooldown=0,
                visible=False  # 隐藏传统渲染
            )
            
            # 将方块添加到面片渲染系统
            import main
            if hasattr(main, 'mesh_renderer') and main.mesh_renderer:
                main.mesh_renderer.add_block(Vec3(*position), id, self._get_block_type(id))

        else:
            # 传统渲染方式
            super().__init__(
                parent=scene,
                position=position,
                model='assets/block',
                origin_y=0.5,
                texture=None,
                scale=0.5,
                color=color.color(0, 0, random.uniform(0.9, 1)),
                collider='box',
                collision_cooldown=0
            )

        
        self.id = id
        # 设置碰撞盒略小于方块视觉大小
        if hasattr(self, 'collider') and self.collider:
            self.collider.scale = Vec3(0.48, 0.48, 0.48)
    
    def _get_block_type(self, block_id):
        """根据方块ID返回方块类型字符串"""
        # 这里可以根据实际的方块ID映射返回相应的类型
        block_types = {
            1: 'grass',
            2: 'dirt', 
            3: 'stone',
            4: 'wood',
            5: 'leaf',
            6: 'bed',  # 基岩
            7: 'brick',
            8: 'check'
        }
        return block_types.get(block_id, 'default')
    
    def destroy(self):
        """销毁方块时同时从面片渲染系统中移除"""
        if Block.use_mesh_splitting_globally:
            import main
            if hasattr(main, 'mesh_renderer') and main.mesh_renderer:
                main.mesh_renderer.remove_block(Vec3(*self.position))
        super().destroy()