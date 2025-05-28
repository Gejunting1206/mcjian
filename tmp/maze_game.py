from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import random
import json
import os
import time
from tkinter.messagebox import *
from threading import Thread
import numpy as np
import sys

app = Ursina()

# 设置窗口
window.title = '迷宫'
window.borderless = True
window.fullscreen = True
window.exit_button.visible = False
window.fps_counter.enabled = True

# 迷宫参数
MAZE_SIZE = 35
WALL_HEIGHT = 3

# 方块贴图列表
Block_list = [
    load_texture('assets/grass_block.png'),
    load_texture('assets/stone_block.png'),
    load_texture('assets/dirt_block.png'),
    load_texture('assets/bed_block.png'),
]

Text.default_font = 'assets/msyh.ttc'

# 在文件顶部添加关卡配置
LEVELS = [
    {"size": 15, "wall_height": 3, "complexity": 0.2},  # 第1关：小迷宫
    {"size": 19, "wall_height": 3, "complexity": 0.3},  # 第2关：中等迷宫
    {"size": 25, "wall_height": 3, "complexity": 0.4},  # 第3关：大迷宫
    {"size": 29, "wall_height": 3, "complexity": 0.5},  # 第4关：更大迷宫
    {"size": 35, "wall_height": 3, "complexity": 0.6},  # 第5关：终极迷宫
]

# 在文件顶部添加
FPS = 30  # 目标帧率
frame_duration = 1.0 / FPS  # 每帧的持续时间


class Block(Button):
    def __init__(self, position=(0, 0, 0), id=0):
        super().__init__(
            parent=scene,
            position=position,
            model='assets/block',
            origin_y=0.5,
            texture=Block_list[id],
            scale=0.5,
            color=color.color(0, 0, random.uniform(0.9, 1))
        )
        self.id = id


class MapEntryType:
    EMPTY = 0
    BLOCK = 1


# 添加自定义加载界面类
class LoadingScreen:
    def __init__(self):
        self.background = Entity(
            parent=camera.ui,
            model='quad',
            scale=(2, 1),
            color=color.black,
            z=-1
        )
        
        self.text_entity = Text(
            parent=camera.ui,
            text='加载中...',
            position=(0, 0),
            origin=(0, 0),
            scale=2,
            color=color.white
        )
        
        self._enabled = True
        self._text = '加载中...'
    
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, value):
        self._text = value
        self.text_entity.text = value
    
    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        self.background.enabled = value
        self.text_entity.enabled = value


class Maze:
    def __init__(self, complexity=0.3):
        self.size = MAZE_SIZE
        self.maze = np.ones((self.size, self.size), dtype=np.int8)
        self.complexity = complexity
        
        # 创建加载界面
        self.loading_screen = LoadingScreen()
        self.loading_screen.text = '正在生成迷宫...'
        
        # 直接生成新迷宫
        self.generate_prim_maze()
        self.create_block_maze()
        self.loading_screen.enabled = False

    def generate_prim_maze(self):
        """使用 Prim 算法生成迷宫"""

        def is_valid(x, y):
            # 确保生成的迷宫在边界内
            return 1 <= x < self.size - 1 and 1 <= y < self.size - 1

        def get_neighbors(x, y, distance=2):
            neighbors = []
            for dx, dy in [(0, distance), (0, -distance), (distance, 0), (-distance, 0)]:
                new_x, new_y = x + dx, y + dy
                if is_valid(new_x, new_y) and self.maze[new_y][new_x] == 1:
                    neighbors.append((new_x, new_y))
            return neighbors

        # 初始化迷宫
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0

        # 墙列表
        walls = []
        walls.extend(get_neighbors(start_x, start_y))

        # 在生成过程中定期更新进度
        total_cells = (self.size - 2) * (self.size - 2)
        cells_processed = 0
        
        while walls:
            # 每处理100个格子更新一次进度
            if cells_processed % 100 == 0:
                self.loading_screen.text = f'生成迷宫... {min(cells_processed/total_cells*100, 99.9):.1f}%'
            # ��机选择一个墙
            wall_x, wall_y = walls.pop(random.randint(0, len(walls) - 1))

            # 检查这个墙的四个方向
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = wall_x + dx, wall_y + dy
                if is_valid(nx, ny) and self.maze[ny][nx] == 0:
                    neighbors.append((nx, ny))

            if neighbors:
                # 随机选择一个已访问的邻居
                px, py = random.choice(neighbors)

                # 打通墙
                self.maze[wall_y][wall_x] = 0
                mid_x = (wall_x + px) // 2
                mid_y = (wall_y + py) // 2
                self.maze[mid_y][mid_x] = 0

                # 添加新的墙
                new_walls = get_neighbors(wall_x, wall_y)
                for w in new_walls:
                    if w not in walls:
                        walls.append(w)

            cells_processed += 1

        # 确保起点和终点可达
        self.maze[1][1] = 0
        self.maze[self.size - 2][self.size - 2] = 0

        # 根据复杂度添加随机通道
        extra_paths = int(self.size * self.complexity)
        for _ in range(extra_paths):
            x = random.randrange(1, self.size - 1, 2)
            y = random.randrange(1, self.size - 1, 2)
            self.maze[y][x] = 0
            if random.random() < 0.5:
                if x + 1 < self.size - 1:
                    self.maze[y][x + 1] = 0
            else:
                if y + 1 < self.size - 1:
                    self.maze[y + 1][x] = 0

    def create_block_maze(self):
        total_blocks = (self.size - 2) * (self.size - 2) + self.size * self.size * (WALL_HEIGHT + 1)
        blocks_created = 0
        
        # 批量创建地板
        floor_blocks = []
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                if (i == 1 and j == 1) or (i == self.size - 2 and j == self.size - 2):
                    continue
                floor_blocks.append((j, 0, i))
                blocks_created += 1
                if blocks_created % 1000 == 0:  # 每1000个方块更新一次进度
                    self.loading_screen.text = f'创建地板... {blocks_created/total_blocks*100:.1f}%'
        
        # 批量创建墙壁
        wall_blocks = []
        for i in range(self.size):
            for j in range(self.size):
                if (i == 0 or i == self.size - 1 or 
                    j == 0 or j == self.size - 1 or 
                    self.maze[i][j] == 1):
                    for h in range(WALL_HEIGHT + 1):
                        block_type = 0 if h == 0 else (2 if h == 1 else 1)
                        wall_blocks.append((j, h, i, block_type))
                        blocks_created += 1
                        if blocks_created % 1000 == 0:
                            self.loading_screen.text = f'创建墙壁... {blocks_created/total_blocks*100:.1f}%'
        
        # 批量实例化方块
        for x, y, z in floor_blocks:
            Block(position=(x, y, z), id=0)
            
        for x, y, z, block_type in wall_blocks:
            Block(position=(x, y, z), id=block_type)


class Game:
    def __init__(self):
        # 从命令行参数获取关卡
        try:
            self.current_level = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        except:
            self.current_level = 0
            
        # 先创建文本显示
        self.level_text = Text(
            text=f'第 {self.current_level + 1} 关',
            position=(-0.85, 0.45),
            scale=1.2,
            color=color.white
        )
        
        self.position_text = Text(
            text='',
            position=(-0.85, 0.4),
            scale=1.2,
            color=color.white
        )
        
        # 然后加载关卡
        self.load_level(self.current_level)

    def load_level(self, level_index):
        """加载指定关卡"""
        if level_index >= len(LEVELS):
            # 通关了
            showinfo('恭喜！', '你已经通关了所有关卡！')
            application.quit()
            return
            
        level_config = LEVELS[level_index]
        global MAZE_SIZE, WALL_HEIGHT
        MAZE_SIZE = level_config["size"]
        WALL_HEIGHT = level_config["wall_height"]
        
        # 清理旧的场景对象
        for entity in scene.entities:
            if not isinstance(entity, (Sky, Text)):  # 保留天空盒和文本
                destroy(entity)
        
        # 更新关卡显示
        self.level_text.text = f'第 {self.current_level + 1} 关'
        
        # 创建新迷宫
        self.maze = Maze(complexity=level_config["complexity"])
        self.player = FirstPersonController(
            position=(1, 5, 1),
            jump_height=1.5
        )
        
        # 创建天空盒和光照
        self.sky = Sky(texture=load_texture('assets/skybox.png'))
        self.light = PointLight(parent=self.player, position=(0, 1, 0), color=color.white)
        AmbientLight(color=color.rgba(0.5, 0.5, 0.5, 0.1))
        
        # 创建起点和终点标记
        self.begin_block = Block(position=(1, 0, 1), id=3)
        self.end_block = Block(position=(self.maze.size - 2, 0, self.maze.size - 2), id=3)
        
        # 重置状态
        self.is_win = False
        self.begin_time = time.time()
        self.path_blocks = []
        self.show_path = False

    def on_win(self):
        if not self.is_win:
            self.end_time = time.time()
            elapsed_time = round(self.end_time - self.begin_time, 2)
            self.is_win = True
            
            # 显示当前关卡完成信息
            if self.current_level < len(LEVELS) - 1:
                if askyesno('过关！', 
                    f'完成第{self.current_level + 1}关！\n'
                    f'用时：{elapsed_time:.2f}秒\n'
                    f'是否进入下一关？'):
                    # 启动下一关
                    os.system(f'python maze_game.py {self.current_level + 1}')
                    application.quit()
            else:
                showinfo('恭喜！', 
                    f'通关！\n'
                    f'最终用时：{elapsed_time:.2f}秒')
                application.quit()

    def find_path(self):
        """使用 A* 算法寻找从玩家当前位置到终点的最短路径"""

        def cal_heuristic(pos, dest):
            """计算启发值 - 曼顿距离"""
            return abs(dest.x - pos[0]) + abs(dest.y - pos[1])

        def get_neighbors(x, y):
            """获取相邻的可行节点"""
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (1 <= new_x < self.maze.size - 1 and
                        1 <= new_y < self.maze.size - 1 and
                        self.maze.maze[new_y][new_x] == 0):
                    neighbors.append((new_x, new_y))
            return neighbors

        # 获取玩家当前位置并转换为网格坐标
        player_x = round(self.player.x)
        player_y = round(self.player.z)  # 注意：在3D空间中z轴对应迷宫y坐标
        start = (player_x, player_y)
        end = (self.maze.size - 2, self.maze.size - 2)

        # 如果玩家位置不合法，返回None
        if not (1 <= player_x < self.maze.size - 1 and
                1 <= player_y < self.maze.size - 1 and
                self.maze.maze[player_y][player_x] == 0):
            return None

        # 初始化和关闭列表
        open_list = {}
        closed_list = {}

        # 将起点加入开启列表
        start_entry = SearchEntry(start[0], start[1], 0.0)
        start_entry.f_cost = cal_heuristic(start, SearchEntry(end[0], end[1], 0.0))
        open_list[start] = start_entry

        while open_list:
            # 获取f值小的节点
            current = min(open_list.values(), key=lambda x: x.f_cost)
            current_pos = current.get_pos()

            # 到终点
            if current_pos == end:
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.pre_entry
                return path[::-1]  # 反转路径

            # 将当前节点移到关闭列表
            open_list.pop(current_pos)
            closed_list[current_pos] = current

            # 检查相邻点
            for next_pos in get_neighbors(current.x, current.y):
                if next_pos in closed_list:
                    continue

                g_cost = current.g_cost + 1  # 邻格子距离为1

                if next_pos not in open_list:
                    # 新节点，加入开启列表
                    next_entry = SearchEntry(next_pos[0], next_pos[1], g_cost)
                    next_entry.f_cost = g_cost + cal_heuristic(next_pos, SearchEntry(end[0], end[1], 0.0))
                    next_entry.pre_entry = current
                    open_list[next_pos] = next_entry
                else:
                    # 已在开启列表，检查是否需要更新
                    next_entry = open_list[next_pos]
                    if g_cost < next_entry.g_cost:
                        next_entry.g_cost = g_cost
                        next_entry.f_cost = g_cost + cal_heuristic(next_pos, SearchEntry(end[0], end[1], 0.0))
                        next_entry.pre_entry = current

        return None  # 没找到路径

    def show_shortest_path(self):
        """显示或隐藏最短路径"""
        if not self.show_path:
            for block in self.path_blocks:
                destroy(block)
            self.path_blocks.clear()
            
            path = self.find_path()
            if path:
                for x, y in path[1:-1]:
                    block = Entity(
                        model='cube',
                        position=(x, 0.3, y),
                        scale=(0.3, 0.05, 0.3),
                        color=Color(random.random(), random.random(), random.random(), 0.8),
                        always_on_top=True
                    )
                    self.path_blocks.append(block)
        else:
            for block in self.path_blocks:
                destroy(block)
            self.path_blocks.clear()
            
        self.show_path = not self.show_path


def input(key):
    if key == 'escape':
        application.quit()
    if key == 'r':  # 重新开始当前关卡
        os.system(f'python maze_game.py {game.current_level}')
        application.quit()
    if key == 'h':
        game.show_shortest_path()


def update():
    start_time = time.time()
    
    # 更新坐标显示
    if hasattr(game, 'player') and hasattr(game, 'position_text'):
        x = int(game.player.x)
        y = int(game.player.y)
        z = int(game.player.z)
        game.position_text.text = f'X:{x} Y:{y} Z:{z}'

    # 检查胜利条件
    if distance_xz(game.player.position, Vec3(game.maze.size - 1, 0, game.maze.size - 1)) < 2:
        if not game.is_win:
            game.end_time = time.time()
        game.on_win()
    
    # 控制帧率
    elapsed_time = time.time() - start_time
    sleep_time = frame_duration - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)


class SearchEntry:
    def __init__(self, x, y, g_cost, f_cost=0, pre_entry=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # 从起点到当前点的代价
        self.f_cost = f_cost  # f = g + h
        self.pre_entry = pre_entry  # 父节点

    def get_pos(self):
        return (self.x, self.y)


# 创建游戏实例
game = Game()

app.run()
