from ursina import *

import ursina
app = ursina.Ursina()
Block_list = [
    load_texture('assets/grass_block.png'),  # 草地方块
    load_texture('assets/stone_block.png'), # 石头方块
    load_texture('assets/dirt_block.png'),  # 泥土方块
    load_texture('assets/bed_block.png'),  # 床方块
    load_texture('assets/log_block.png'),   # 原木方块
    load_texture('assets/leaf_block.png')   # 树叶方块
]
# 物品栏系统
from ursina import *
from block import Block  # 导入Block类

class Hotbar:
    """物品栏类，用于显示和管理玩家可用的方块类型"""
    
    def __init__(self, block_textures, parent=camera.ui):
        self.parent = parent
        self.block_textures = block_textures  # 方块纹理列表
        self.num_slots = min(9, len(block_textures))  # 物品栏槽位数量，最多9个
        self.selected_slot = 0  # 当前选中的槽位
        
        # 加载物品栏背景纹理
        self.background_texture = load_texture('assets/hotbar.png')
        
        # 创建物品栏背景
        self.background = Entity(
            parent=self.parent,
            model='quad',
            texture=self.background_texture,
            scale=(0.8, 0.1),
            position=(0, -0.45),  # 屏幕底部居中
            color=color.white,
        )
        
        # 创建选中框
        self.selection_highlight = Entity(
            parent=self.parent,
            model='quad',
            texture='assets/check_block.png',
            scale=(0.08, 0.08),
            position=self._get_slot_position(self.selected_slot)
        )
        
        # 创建物品图标 - 使用3D方块而不是纹理
        self.item_icons = []
        for i in range(self.num_slots):
            # 创建微型Block实例作为UI元素
            icon = Entity(
                parent=self.parent,
                model='block',
                texture=self.block_textures[i],
                scale=(0.02, 0.02, 0.02),  # 增大方块大小
                position=self._get_slot_position(i),  # 稍微上移以避免与背景重叠
                rotation=(-25, -45, -25),  # 调整旋转角度以更好地展示方块
                visible=True  # 默认可见
            )
            self.item_icons.append(icon)
    
    def _get_slot_position(self, slot_index):
        """计算指定槽位的屏幕位置"""
        # 计算起始x坐标（最左侧槽位）
        start_x = -0.35
        # 槽位间距
        slot_spacing = 0.0865
        # 计算x坐标
        x = start_x + slot_index * slot_spacing
        # y坐标固定在底部
        y = -0.45
        return Vec3(x, y, 0)  # 返回Vec3而不是元组，以便与Vec3相加
    
    def update_selection(self, slot_index):
        """更新选中的槽位"""
        if 0 <= slot_index < self.num_slots:
            self.selected_slot = slot_index
            # 更新选中框位置
            self.selection_highlight.position = self._get_slot_position(self.selected_slot)
            return True
        return False
    
    def get_selected_block_id(self):
        """获取当前选中的方块ID"""
        return self.selected_slot
    
    def input(self, key):
        """处理键盘输入，切换选中的槽位"""
        # 数字键1-9选择对应槽位
        if key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            slot = int(key) - 1  # 转换为0-8的索引
            if slot < self.num_slots:
                self.update_selection(slot)
                return True
        
        # 鼠标滚轮切换槽位
        if key == 'scroll up':
            new_slot = (self.selected_slot - 1) % self.num_slots
            self.update_selection(new_slot)
            return True
        elif key == 'scroll down':
            new_slot = (self.selected_slot + 1) % self.num_slots
            self.update_selection(new_slot)
            return True
        
        return False
Hotbar(Block_list)

app.run()