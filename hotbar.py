# 物品栏系统
from ursina import *
from block import Block  # 导入Block类

class Hotbar:
    """物品栏类，用于显示和管理玩家可用的方块类型"""
    
    def __init__(self, block_textures, parent=camera.ui):
        self.parent = parent
        self.block_textures = block_textures  # 方块纹理列表
        self.max_slots = 9  # 物品栏最大槽位数量
        self.selected_slot = 0  # 当前选中的槽位
        
        # 初始化已收集的方块列表和对应的ID映射
        self.collected_blocks = []  # 存储已收集的方块ID
        self.block_id_to_slot = {}  # 方块ID到槽位的映射
        
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
            position=self._get_slot_position(0)  # 初始位置在第一个槽位
        )
        
        # 创建物品图标 - 初始为空
        self.item_icons = []
    
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
        if 0 <= slot_index < len(self.collected_blocks):
            self.selected_slot = slot_index
            # 更新选中框位置
            self.selection_highlight.position = self._get_slot_position(self.selected_slot)
            return True
        return False
    
    def get_selected_block_id(self):
        """获取当前选中的方块ID"""
        if not self.collected_blocks:
            return None  # 如果没有收集任何方块，返回None
        return self.collected_blocks[self.selected_slot]
    
    def input(self, key):
        """处理键盘输入，切换选中的槽位"""
        # 如果物品栏为空，则不处理任何输入
        if not self.collected_blocks:
            return False
            
        # 数字键1-9选择对应槽位
        if key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            slot = int(key) - 1  # 转换为0-8的索引
            if slot < len(self.collected_blocks):
                self.update_selection(slot)
                return True
        
        # 鼠标滚轮切换槽位
        if key == 'scroll up':
            if len(self.collected_blocks) > 0:
                new_slot = (self.selected_slot - 1) % len(self.collected_blocks)
                self.update_selection(new_slot)
                return True
        elif key == 'scroll down':
            if len(self.collected_blocks) > 0:
                new_slot = (self.selected_slot + 1) % len(self.collected_blocks)
                self.update_selection(new_slot)
                return True
        
        return False
        
    def collect_block(self, block_id):
        """使用鼠标中键收集方块
        
        这个方法应该在游戏主循环中调用，当玩家对方块点击鼠标中键时
        
        Args:
            block_id: 要收集的方块ID
            
        Returns:
            bool: 是否成功收集方块（如果已存在则选中并返回False）
        """
        return self.add_block(block_id)
        
    def add_block(self, block_id):
        """添加方块到物品栏或选中已有方块
        
        Args:
            block_id: 要添加的方块ID
            
        Returns:
            bool: 是否成功添加方块（如果已存在则选中并返回False）
        """
        # 检查方块是否已经在物品栏中
        if block_id in self.block_id_to_slot:
            # 如果已存在，选中该方块
            slot = self.block_id_to_slot[block_id]
            self.update_selection(slot)
            return False
        
        # 检查物品栏是否已满
        if len(self.collected_blocks) >= self.max_slots:
            # 如果已满，替换当前选中的方块
            old_block_id = self.collected_blocks[self.selected_slot]
            # 更新映射
            del self.block_id_to_slot[old_block_id]
            self.collected_blocks[self.selected_slot] = block_id
            self.block_id_to_slot[block_id] = self.selected_slot
            
            # 更新图标
            if self.selected_slot < len(self.item_icons):
                # 更新现有图标
                self.item_icons[self.selected_slot].texture = self.block_textures[block_id]
            else:
                # 创建新图标
                icon = Entity(
                    parent=self.parent,
                    model='block',
                    texture=self.block_textures[block_id],
                    scale=(0.02, 0.02, 0.02),  # 增大方块大小
                    position=self._get_slot_position(self.selected_slot),
                    rotation=(-25, -45, -25),  # 调整旋转角度以更好地展示方块
                    visible=True  # 默认可见
                )
                self.item_icons.append(icon)
        else:
            # 添加新方块到物品栏
            slot = len(self.collected_blocks)
            self.collected_blocks.append(block_id)
            self.block_id_to_slot[block_id] = slot
            
            # 创建新图标
            icon = Entity(
                parent=self.parent,
                model='block',
                texture=self.block_textures[block_id],
                scale=(0.02, 0.02, 0.02),  # 增大方块大小
                position=self._get_slot_position(slot),
                rotation=(-25, -45, -25),  # 调整旋转角度以更好地展示方块
                visible=True  # 默认可见
            )
            self.item_icons.append(icon)
            
            # 如果这是第一个方块，选中它
            if len(self.collected_blocks) == 1:
                self.update_selection(0)
        
        return True