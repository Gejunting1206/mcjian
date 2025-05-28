# 方块类定义
from ursina import *

class Block(Button):
    def __init__(self, position=(0, 0, 0), id=0):
        super().__init__(
            parent=scene,
            position=position,
            model='assets/block',
            origin_y=0.5,
            texture=None,
            scale=0.5,
            color=color.color(0, 0, random.uniform(0.9, 1)),
            # 调整碰撞盒大小，稍微缩小以减少边缘碰撞导致的弹跳
            collider='box',
            collision_cooldown=0
        )
        self.id = id
        # 设置碰撞盒略小于方块视觉大小
        if hasattr(self, 'collider') and self.collider:
            self.collider.scale = Vec3(0.48, 0.48, 0.48)