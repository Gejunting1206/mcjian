# 游戏引擎和核心功能模块
from ursina import *  # Ursina游戏引擎核心
from ursina.prefabs.first_person_controller import FirstPersonController  # 第一人称控制器

# 地形生成相关
from perlin_noise import PerlinNoise  # 柏林噪声生成地形

# 屏幕截图功能
from PIL import ImageGrab  # 用于获取屏幕尺寸

# 数学和物理计算
from ursina.ursinamath import distance  # 距离计算
from math import floor, sin, cos, pi  # 基础数学函数

# 多线程和并发
from threading import Thread, Lock  # 线程和锁
from concurrent.futures import ThreadPoolExecutor  # 线程池

# 实用工具
import logging # 日志记录
import os  # 操作系统接口
import gc  # 垃圾回收
import time  # 时间相关功能
import numpy as np  # 数值计算

# 随机数生成
from random import randint, choice, uniform
import random  # 随机数生成器

# Ursina组件
from ursina.prefabs.health_bar import HealthBar  # 血条UI
from ursina.models.procedural.quad import Quad  # 四边形模型
from ursina.vec3 import Vec3  # 3D向量
from ursina import Sprite, Text  # 精灵和文本

# 数据结构
from collections import deque  # 双端队列，用于对象池
from queue import PriorityQueue, Empty # 优先级队列

# 性能优化模块
from optimization.frustum_culling import frustum_culling_manager, get_visible_blocks  # 视锥体剔除
from lod_system import lod_manager, process_blocks_lod  # LOD系统
from optimization.performance_optimizer import performance_optimizer, handle_optimization_hotkeys  # 性能优化管理器
from performance_config import perf_config, PerformanceConfig  # 性能配置管理
# 导入输入优化器
from input_optimizer import input_optimizer
# 区块加载优化模块
from loading_system import chunk_loader, mark_chunk_loaded  # 区块加载系统
from block_cache import block_cache  # 区块缓存系统
from chunk_loading_optimizer import chunk_loading_optimizer, preload_initial_chunks, integrate_with_game_loop  # 区块加载优化器
# 渐进式加载系统 - 解决游戏启动时帧率波动问题
from progressive_loading import progressive_loader, integrate_with_game_loop as progressive_loading_update, initialize_on_game_start  # 渐进式加载系统

# GPU优化相关模块
from gpu.gpu_frustum_culling import gpu_frustum_culling  # GPU视锥体剔除
from gpu.gpu_optimization_manager import gpu_optimization_manager  # GPU优化管理器
from gpu.gpu_optimization_integration import gpu_optimization_integrator  # GPU优化集成器

# 导入集成优化管理器
from optimization.optimization_integration import OptimizationManager  # 集成优化管理器

# 导入综合性能优化器
from optimization.comprehensive_performance_optimizer import comprehensive_optimizer  # 综合性能优化器

# 导入面片渲染系统
from mesh_splitting_renderer import mesh_renderer, MeshSplittingRenderer

# 高效渲染优化器
class HighPerformanceRenderer:
    """高性能渲染器 - 专注于最大化帧率"""
    
    def __init__(self):
        self.visible_objects = set()
        self.render_cache = {}
        self.last_camera_pos = None
        self.camera_move_threshold = 2.0  # 降低阈值以减少单次更新负载
        self.frame_skip_counter = 0
        
    def should_update_visibility(self, camera_pos):
        """判断是否需要更新可见性"""
        if self.last_camera_pos is None:
            self.last_camera_pos = camera_pos
            return True
        
        # 只有摄像机移动超过阈值才更新
        distance = ((camera_pos.x - self.last_camera_pos.x) ** 2 + 
                   (camera_pos.y - self.last_camera_pos.y) ** 2 + 
                   (camera_pos.z - self.last_camera_pos.z) ** 2) ** 0.5
        
        if distance > self.camera_move_threshold:
            self.last_camera_pos = camera_pos
            return True
        return False
    
    def fast_render(self, camera_pos):
        """快速渲染 - 跳帧渲染"""
        self.frame_skip_counter += 1
        
        # 每6帧才执行一次完整渲染
        if self.frame_skip_counter % 6 == 0:
            # 简化的渲染逻辑
            if self.should_update_visibility(camera_pos):
                # 更新可见对象列表
                self._update_visible_objects(camera_pos)
            
            # 渲染可见对象
            self._render_visible_objects()
    
    def _update_visible_objects(self, camera_pos):
        """更新可见对象列表"""
        # 使用空间哈希快速查找附近对象
        nearby_objects = spatial_hash.query_nearby(camera_pos, radius=1)  # 进一步减小查询半径至当前区块，修复float转int错误
        
        # 简单的距离剔除
        self.visible_objects.clear()
        max_objects_per_frame = 100  # 限制每帧处理对象数量
        processed_count = 0
        for obj in nearby_objects:
            if processed_count >= max_objects_per_frame:
                break
            if hasattr(obj, 'position'):
                distance_sq = ((obj.position.x - camera_pos.x) ** 2 + 
                              (obj.position.y - camera_pos.y) ** 2 + 
                              (obj.position.z - camera_pos.z) ** 2)
                # 只处理近距离对象
                if distance_sq < 25:  # 只处理5个单位内的对象
                    self.visible_objects.add(obj)
    
    def _render_visible_objects(self):
        """渲染可见对象"""
        # 简化的批量渲染
        for obj in self.visible_objects:
            if hasattr(obj, 'enabled') and obj.enabled:
                # 添加到批量渲染器
                material_key = getattr(obj, 'texture', 'default')
                batch_renderer.add_to_batch(obj, material_key)

# 创建高性能渲染器实例
high_perf_renderer = HighPerformanceRenderer()

# 设置默认字体为支持中文的Minecraft字体
Text.default_font = 'assets/5_Minecraft AE(支持中文).ttf'

# 算法级性能优化 - 实现空间哈希和批量渲染
class SpatialHashGrid:
    """空间哈希网格 - 用于快速空间查询和碰撞检测"""
    
    def __init__(self, cell_size=32):
        self.cell_size = cell_size
        self.grid = {}
        self.object_to_cells = {}  # 对象到网格单元的映射
    
    def _hash_position(self, position):
        """将3D位置哈希到网格坐标"""
        return (
            int(position.x // self.cell_size),
            int(position.y // self.cell_size), 
            int(position.z // self.cell_size)
        )
    
    def insert(self, obj, position):
        """插入对象到空间哈希网格"""
        cell = self._hash_position(position)
        if cell not in self.grid:
            self.grid[cell] = set()
        self.grid[cell].add(obj)
        self.object_to_cells[obj] = cell
    
    def remove(self, obj):
        """从空间哈希网格移除对象"""
        if obj in self.object_to_cells:
            cell = self.object_to_cells[obj]
            if cell in self.grid:
                self.grid[cell].discard(obj)
                if not self.grid[cell]:  # 如果网格单元为空，删除它
                    del self.grid[cell]
            del self.object_to_cells[obj]
    
    def query_nearby(self, position, radius=1):
        """查询附近的对象"""
        center_cell = self._hash_position(position)
        nearby_objects = set()
        
        # 检查周围的网格单元
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    cell = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                    if cell in self.grid:
                        nearby_objects.update(self.grid[cell])
        
        return nearby_objects

# 批量渲染管理器
class BatchRenderManager:
    """批量渲染管理器 - 减少渲染调用次数"""
    
    def __init__(self):
        self.batches = {}  # 按材质/纹理分组的批次
        self.dirty_batches = set()  # 需要更新的批次
        self.max_batch_size = 500  # 每个批次最大对象数
    
    def add_to_batch(self, obj, material_key):
        """将对象添加到批次"""
        if material_key not in self.batches:
            self.batches[material_key] = []
        
        batch = self.batches[material_key]
        if len(batch) < self.max_batch_size:
            batch.append(obj)
            self.dirty_batches.add(material_key)
    
    def render_batches(self, camera_position):
        """渲染所有批次"""
        rendered_objects = 0
        
        for material_key in list(self.dirty_batches):
            if material_key in self.batches:
                batch = self.batches[material_key]
                if batch:
                    # 距离排序优化 - 只对前50个对象排序
                    if len(batch) > 50:
                        batch = batch[:50]
                    
                    rendered_objects += len(batch)
        
        self.dirty_batches.clear()
        return rendered_objects

# 异步任务管理器
class AsyncTaskManager:
    """异步任务管理器 - 将耗时操作移到后台线程"""
    
    def __init__(self, max_workers=1):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks = {}
    
    def submit_task(self, task_id, func, *args, **kwargs):
        """提交异步任务"""
        if task_id not in self.pending_tasks:
            future = self.executor.submit(func, *args, **kwargs)
            self.pending_tasks[task_id] = future
    
    def check_completed_tasks(self):
        """检查已完成的任务"""
        completed = []
        for task_id, future in list(self.pending_tasks.items()):
            if future.done():
                try:
                    result = future.result()
                    completed.append((task_id, result))
                except Exception:
                    completed.append((task_id, None))
                del self.pending_tasks[task_id]
        return completed

# 创建全局优化实例
spatial_hash = SpatialHashGrid(cell_size=32)
batch_renderer = BatchRenderManager()
async_manager = AsyncTaskManager(max_workers=1)

# 添加帧率控制 - 极限优化
FPS = 200  # 极限目标帧率 (从120提高到200)
frame_duration = 1.0 / FPS  # 每帧的持续时间
MAX_FRAME_TIME = 1.0 / 150  # 最大帧时间，确保至少150FPS

# 动态渲染距离参数 - 极限优化
DYNAMIC_RENDER_DISTANCE = True  # 启用动态渲染距离
MIN_RENDER_DISTANCE = 0         # 最小渲染距离 (从1降低到0)
MAX_RENDER_DISTANCE = 1         # 最大渲染距离 (保持1)
RENDER_DISTANCE_UPDATE_INTERVAL = 0.2  # 渲染距离更新间隔（秒）(从0.5减小到0.2)

# 空间网格配置 - 极限优化
GRID_CELL_SIZE = 64 # 进一步增加网格单元大小以减少计算量 (从48增加到64)

# 创建优化管理器
optimization_manager = OptimizationManager(chunk_size=GRID_CELL_SIZE, render_distance=MAX_RENDER_DISTANCE)

# 优化管理器配置 - 极限优化
optimization_manager.use_instanced_rendering = True
optimization_manager.use_mesh_combining = True
optimization_manager.use_distance_culling = True
optimization_manager.use_chunk_management = True
optimization_manager.adaptive_optimization = True
optimization_manager.target_fps = 300  # 极限目标帧率 (从60提高到300)

# 配置综合性能优化器 - 极限优化
comprehensive_optimizer.target_fps = 300  # 极限目标帧率 (从150提高到300)
comprehensive_optimizer.min_acceptable_fps = 200  # 极限最低可接受帧率 (从100提高到200)
comprehensive_optimizer.adaptive_mode = True  # 启用自适应模式
comprehensive_optimizer.optimization_level = 15  # 极限优化级别 (从6提高到15)

# 区块加载参数 - 极限优化
MAX_CHUNKS_PER_UPDATE = 1  # 极限减少每次更新最大加载区块数 (保持1)
CHUNK_LOAD_INTERVAL = 1.0  # 区块加载间隔时间 (从0.5增加到1.0)

# 性能统计显示相关变量
# 在文件开头添加全局变量声明
global hand, sky, fps_text, performance_stats_text, hotbar

# 导入共享类
from block import Block  # 方块类

# 导入物品栏系统
from hotbar import Hotbar  # 物品栏类

show_performance_stats = False  # 是否显示性能统计
performance_stats_text = None  # 性能统计文本对象

# 游戏状态枚举
class GameState:
    """
    游戏状态枚举类
    定义游戏的不同状态:
    - MAIN_MENU: 主菜单状态
    - PLAYING: 游戏进行中状态
    """
    MAIN_MENU = 0  # 主菜单状态
    PLAYING = 1    # 游戏进行中状态

# 当前游戏状态，初始为主菜单
current_state = GameState.MAIN_MENU

# 初始化Ursina游戏引擎
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Ursina()

# 获取屏幕尺寸并设置窗口属性
size = ImageGrab.grab().size  # 获取当前屏幕分辨率
window.fps_counter.enabled = True   # 显示帧率计数器
window.exit_button.visible = False  # 隐藏默认退出按钮
window.title = '3D Game'           # 窗口标题
window.fullscreen = False          # 非全屏模式
window.size = (int(size[0]), int(size[1]))  # 设置窗口大小为屏幕分辨率
window.position = (0, 0)           # 窗口位置在屏幕左上角

# 添加帧率计算相关变量
frame_count = 0
last_time = time.time()
fps = 0
fps_text = None  # 初始化fps_text变量

# UI隐藏状态变量
ui_hidden = False

# 区块生成相关配置
# 线程池用于异步生成区块，max_workers控制最大并发线程数
# 优化线程池配置，减少资源竞争
num_cores = os.cpu_count() or 1 # 获取CPU核心数，如果获取失败则默认为1
chunk_executor = ThreadPoolExecutor(max_workers=max(1, num_cores - 1))  # 进一步减少线程数量，避免资源竞争

# 线程锁，确保多线程安全
chunk_lock = Lock()

# 区块生成性能保护参数
sync_chunk_generation_counter = 0  # 同步区块生成计数器
MAX_SYNC_CHUNKS_PER_FRAME = 1  # 每帧最多同步生成的区块数量 (从2降低到1)
MIN_FPS_FOR_SYNC_GENERATION = 30  # 低于此帧率时禁用同步生成 (从15提高到30)

# 柏林噪声生成器配置
# octaves: 噪声层数，影响地形细节
# seed: 随机种子，基于当前时间生成
noise = PerlinNoise(octaves=3, seed=int(time.time()))

# 地形生成参数
scale = 50        # 地形缩放系数
height_scale = 10 # 高度缩放系数
base_height = 5   # 基础高度
# 水位相关代码已移除

# 添加区块预加载队列 - 用于优先加载玩家下方区块
# preload_queue = []  # 不再需要，使用 chunk_load_queue 代替

# 玩家变量，将在游戏开始时初始化
player = None

# 初始化面片渲染器
mesh_renderer = None

would = []

# 方块纹理列表
# 包含游戏中所有方块的顶部/侧面纹理
Block_list = [
    load_texture('assets/grass_block.png'),  # 草地方块
    load_texture('assets/stone_block.png'), # 石头方块
    load_texture('assets/dirt_block.png'),  # 泥土方块
    load_texture('assets/bed_block.png'),  # 床方块
    load_texture('assets/log_block.png'),   # 原木方块
    load_texture('assets/leaf_block.png')   # 树叶方块
]

# 方块ID定义
GRASS = 0
STONE = 1
DIRT = 2
BED = 3
LOG = 4
LEAF = 5

# 方块破坏效果纹理列表
# 包含10个不同阶段的破坏效果纹理
crack_textures = [load_texture(f'assets/crack_{i}.png') for i in range(1, 11)]

block_type_id = 0
# 游戏资源加载
# 天空盒纹理 - 用于背景
sky_texture = load_texture('assets/skybox.png')
# 手臂纹理 - 第一人称视角显示
arm_texture = load_texture('assets/arm_texture.png')
# 破坏方块音效
punch_sound = Audio(sound_file_name='assets/punch_sound.wav', loop=False, autoplay=False)
# 物品栏实例
hotbar = None  # 将在游戏开始时初始化


# 空间网格类定义

class SpatialGrid:
    def __init__(self):
        self.grid = {}
        self.last_frustum_update = 0
        self.frustum_update_interval = 0.1  # 减少视锥体更新间隔以提高剔除效率 (从0.8减小到0.1)
        self.last_cleanup_time = 0
        self.cleanup_interval = 5.0  # 更频繁地清理未使用的网格单元

    def _get_cell_coords(self, position):
        # 根据方块位置计算其所在的网格单元坐标
        return (
            floor(position.x / GRID_CELL_SIZE),
            floor(position.y / GRID_CELL_SIZE),
            floor(position.z / GRID_CELL_SIZE)
        )

    def add_block(self, block):
        cell_coords = self._get_cell_coords(block.position)
        if cell_coords not in self.grid:
            self.grid[cell_coords] = []
        # 避免重复添加
        if block not in self.grid[cell_coords]:
            self.grid[cell_coords].append(block)

    def remove_block(self, block):
        cell_coords = self._get_cell_coords(block.position)
        if cell_coords in self.grid and block in self.grid[cell_coords]:
            self.grid[cell_coords].remove(block)
            # 如果单元格变空，可以选择性地删除该键以节省内存
            if not self.grid[cell_coords]:
                del self.grid[cell_coords]

    def get_nearby_blocks(self, position, radius=1):
        # 获取指定位置附近单元格中的所有方块
        nearby_blocks = set() # 使用集合避免重复
        center_cell = self._get_cell_coords(position)
        cx, cy, cz = center_cell
        
        # 优化循环 - 根据当前帧率动态调整搜索半径
        from ursina import application
        current_fps = getattr(application, 'fps', 30)
        
        # 在低帧率下减小搜索半径
        if current_fps < 15:
            radius = max(1, radius - 1)  # 减小搜索半径
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # 跳过对角线上的单元格，减少检查的单元格数量
                    if abs(dx) == radius and abs(dy) == radius and abs(dz) == radius:
                        continue
                        
                    cell_coords = (cx + dx, cy + dy, cz + dz)
                    if cell_coords in self.grid:
                        # 添加该单元格中的所有方块到集合中
                        nearby_blocks.update(self.grid[cell_coords])
        
        # 定期清理未使用的网格单元
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.last_cleanup_time = current_time
            self._cleanup_unused_cells(position, radius * 3)  # 清理更大范围外的单元格
        
        # 返回列表
        return list(nearby_blocks)
    
    def get_visible_blocks(self, position, radius=1):
        """获取视锥体内可见的方块，并应用LOD"""
        # 获取附近方块
        visible_blocks = []
        blocks = self.get_nearby_blocks(position, radius)
        
        # 应用LOD系统 - 根据距离设置不同的细节级别
        for block in blocks:
            # 计算与玩家的距离
            dist = distance(block.position, position)
            
            # 根据距离设置LOD级别
            if dist > 20:
                block.lod_level = 2  # 低细节
            elif dist > 10:
                block.lod_level = 1  # 中等细节
            else:
                block.lod_level = 0  # 高细节
                
            visible_blocks.append(block)
        
        # 使用性能优化管理器进行视锥体剔除
        return performance_optimizer.get_visible_blocks(self, position, radius, visible_blocks)
        
    def _cleanup_unused_cells(self, position, max_distance):
        """清理远离玩家的未使用网格单元以节省内存"""
        if not self.grid:
            return
            
        center_cell = self._get_cell_coords(position)
        cells_to_remove = []
        
        # 找出距离过远的单元格
        for cell_coords in self.grid.keys():
            dx = abs(cell_coords[0] - center_cell[0])
            dy = abs(cell_coords[1] - center_cell[1])
            dz = abs(cell_coords[2] - center_cell[2])
            
            # 计算曼哈顿距离
            distance = dx + dy + dz
            
            if distance > max_distance:
                cells_to_remove.append(cell_coords)
        
        # 移除远处的单元格
        for cell_coords in cells_to_remove:
            if cell_coords in self.grid:
                del self.grid[cell_coords]

# 全局空间网格实例
spatial_grid = SpatialGrid()

# 区块管理系统配置
# CHUNK_SIZE: 每个区块的大小(以方块为单位)
CHUNK_SIZE = 4
# RENDER_DISTANCE: 渲染距离(以区块为单位)
RENDER_DISTANCE = 1  # 初始渲染距离，将由动态调整系统控制
# MAX_CHUNKS_PER_UPDATE: 每帧最大生成区块数量
MAX_CHUNKS_PER_UPDATE = 2  # 适当提高每帧生成的区块数量
# UPDATE_INTERVAL: 区块更新间隔(秒)
UPDATE_INTERVAL = 0.3  # 缩短更新间隔以提高响应性
# CHUNK_UNLOAD_DISTANCE: 区块卸载距离(以区块为单位)
CHUNK_UNLOAD_DISTANCE = 4  # 更积极地卸载远处区块
# loaded_chunks: 已加载区块字典，键为区块坐标，值为区块对象
loaded_chunks = {}

# 上次渲染距离更新时间
last_render_distance_update = 0

# 区块加载优先级队列
# 存储待加载区块及其优先级，确保重要区块优先加载
chunk_load_queue = PriorityQueue()

def get_spiral_sequence(center, max_distance):
    """
    生成以给定中心点为中心的螺旋坐标序列
    
    参数:
        center (tuple): 中心点坐标(x,z)
        max_distance (int): 最大螺旋半径
    
    返回:
        list: 按螺旋顺序排列的坐标列表，从中心向外扩展
    
    算法说明:
        1. 从中心点开始
        2. 按右->上->左->下的顺序向外螺旋扩展
        3. 每圈增加移动步数
    """
    result = []
    x, z = center
    
    # 添加中心点
    result.append((x, z))
    
    # 生成螺旋序列
    for d in range(1, max_distance + 1):
        # 向右移动d步
        for i in range(d):
            x += 1
            result.append((x, z))
        
        # 向上移动d步
        for i in range(d):
            z += 1
            result.append((x, z))
        
        # 向左移动d+1步
        for i in range(d + 1):
            x -= 1
            result.append((x, z))
        
        # 向下移动d+1步
        for i in range(d + 1):
            z -= 1
            result.append((x, z))
        
        # 向右移动1步(为下一圈做准备)
        for i in range(1):
            x += 1
            result.append((x, z))
    
    return result

# 创建粒子对象池
MAX_PARTICLES = 30  # 进一步减少最大粒子数量以降低内存使用
particle_pool = deque(maxlen=MAX_PARTICLES)

# 水相关功能已移除

class Block(Button):
    def __init__(self, position=(0, 0, 0), id=0):
        # 优化方块初始化
        super().__init__(
            parent=scene,
            position=position,
            model='assets/block.obj',  # 所有方块使用相同的模型
            origin_y=0.5,
            texture=Block_list[id],
            scale=0.5,
            color=color.white,
            collision=True,
            collider='box',  # 使用简单的碰撞体积
        )
        self.id = id
        self.texture_scale = (1, 1)
        self.always_on_top = False
        
        # 添加距离检查优化
        self.last_update = 0
        self.last_distance_check = 0
        self.in_range = False  # 是否在交互范围内
        self.distance_to_player = 999  # 初始距离设为很大的值
        
        # 优化渲染 - 远处的方块可以降低更新频率
        self.update_interval = 0.1  # 默认更新间隔
        
        # 减少碰撞检测频率
        self.collision_cooldown = 0
        
        # LOD系统支持
        self.lod_level = 0  # 默认使用最高细节级别
        self.visible_in_frustum = True  # 是否在视锥体内可见

    def input(self, key):
        # 大幅优化交互检测 - 减少距离计算和碰撞检测频率
        # 只处理鼠标按键事件，忽略其他输入事件
        if not (key == 'right mouse down' or key == 'left mouse down' or key == 'middle mouse down'):
            return
            
        current_time = time.time()
        
        # 只有当鼠标悬停在方块上时才进行进一步处理
        if mouse.hovered_entity == self:  # 使用mouse.hovered_entity替代self.hovered
            # 进一步减少距离计算频率 - 每0.2秒最多计算一次
            if current_time - self.last_distance_check > 0.2:  # 从0.1增加到0.2
                self.last_distance_check = current_time
                if hasattr(player, 'position'):
                    self.distance_to_player = distance(player, self)
                    self.in_range = 0 <= self.distance_to_player <= 5
            
            # 使用缓存的距离值进行判断
            # 增加冷却时间检查，并根据距离动态调整处理频率
            # 距离越远，处理频率越低
            should_process = self.collision_cooldown <= 0 and self.in_range
            if should_process and self.distance_to_player > 3: # 对于距离稍远的方块 (3到5之间)
                # 随机跳过一部分处理，降低频率
                if random.random() > 0.5: # 50% 的概率跳过
                    should_process = False
                    
            if should_process:
                # 播放音效
                punch_sound.play()
            
            if key == 'right mouse down':
                # 设置冷却时间
                self.collision_cooldown = 5  # 仅在放置方块时设置冷却时间
                    # 放置方块
                    # 确保放置的方块在玩家可及范围内
                if distance(player.position, self.position) < 6:
                        # 放置方块
                        # 确保新方块的位置是整数坐标
                        pos = self.position + mouse.normal
                        pos = Vec3(floor(pos.x + 0.5), floor(pos.y + 0.5), floor(pos.z + 0.5))
                        chunk_pos = get_chunk_position(pos)
                    
                        # 优化方块存在检查 - 使用哈希表加速查找
                        if chunk_pos in loaded_chunks:
                            # 使用更高效的方式检查位置是否已有方块
                            # 创建一个集合而不是字典，减少内存使用
                            block_positions = set()
                            for block in loaded_chunks[chunk_pos].blocks:
                                block_positions.add(str(block.position))
                        
                        # 使用空间网格检查目标位置是否已有方块
                        # 获取目标位置的单元格坐标
                        target_cell_coords = spatial_grid._get_cell_coords(pos)
                        occupied = False
                        # 优化：只检查目标单元格（方块不会跨越单元格边界）
                        check_cell = target_cell_coords
                        if check_cell in spatial_grid.grid:
                            for existing_block in spatial_grid.grid[check_cell]:
                                # 检查精确位置是否重叠
                                if existing_block.position == pos:
                                     occupied = True
                                     break

                        # if str(pos) not in block_positions and chunk_pos in loaded_chunks: # 旧的检查方式
                        if not occupied and chunk_pos in loaded_chunks: # 使用空间网格检查结果
                            # 尝试从对象池获取方块
                            new_block = None
                            if Chunk._block_pool and len(Chunk._block_pool) > 0:
                                try:
                                    with chunk_lock:
                                        if Chunk._block_pool:  # 再次检查，确保池不为空
                                            new_block = Chunk._block_pool.pop()
                                    
                                    # 重置方块属性
                                    if new_block:
                                        new_block.position = pos
                                        new_block.id = block_type_id
                                        new_block.texture = Block_list[block_type_id]
                                        new_block.visible = True
                                        new_block.collision_cooldown = 0
                                        new_block.last_update = 0
                                        new_block.last_distance_check = 0
                                        new_block.in_range = False
                                        new_block.distance_to_player = 999
                                except Exception:
                                    new_block = None
                            
                            # 如果没有从对象池获取到方块，创建新方块
                            if new_block is None:
                                new_block = Block(position=pos, id=block_type_id)
                            # 水方块相关代码已移除
                            
                            # 添加到区块
                            loaded_chunks[chunk_pos].blocks.append(new_block)
                            # 添加到空间网格
                            spatial_grid.add_block(new_block)
                            # 添加到优化管理器
                            optimization_manager.add_block(new_block)
                
            elif key == 'left mouse down':
                    # 破坏方块
                    chunk_pos = get_chunk_position(self.position)
                    if chunk_pos in loaded_chunks:
                        if self in loaded_chunks[chunk_pos].blocks:  # 检查方块是否存在
                            # 从空间网格移除
                            spatial_grid.remove_block(self)
                            # 从区块列表移除
                            loaded_chunks[chunk_pos].blocks.remove(self)
                            # 从优化管理器移除
                            optimization_manager.remove_block(self)
                            # 立即更新玩家重力状态
                            if player and hasattr(player, 'gravity'):
                                player.gravity = 1.0  # 默认重力值
                                if hasattr(player, 'current_gravity'):
                                    player.current_gravity = 1.0
                            
                            # 尝试将方块添加到对象池而不是销毁
                            try:
                                with chunk_lock:
                                    if len(Chunk._block_pool) < Chunk._block_pool.maxlen:
                                        # 隐藏方块但不销毁
                                        self.visible = False
                                        # 将方块添加到对象池
                                        Chunk._block_pool.append(self)
                                        return  # 已添加到对象池，不需要销毁
                            except Exception:
                                pass  # 如果添加到对象池失败，继续销毁
                            
                            # 如果没有添加到对象池，销毁方块
                            destroy(self)
                
            elif key == 'middle mouse down':
                    # 获取点击的方块类型
                    if hasattr(self, 'id') and hotbar:
                        # 调用物品栏的收集方块方法
                        hotbar.collect_block(self.id)
        
        # 更新冷却时间
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1

    




class Chunk:
    # 类级别的对象池，用于重用方块对象
    # 减小对象池大小，避免占用过多内存，但保持足够大以提高重用率
    _block_pool = deque(maxlen=1000)  # 增加对象池大小以减少新建开销
    
    def __init__(self, position):
        self.position = position
        self.blocks = []
        self.is_generated = False
        self._future = None
        
        # 减少日志输出，降低I/O开销
        # print(f"Chunk generated at {position}")
        
        self.generate()

    def _finalize_chunk_creation(self, blocks_data, use_batching=False):
        """处理区块数据的最终创建逻辑，供同步和异步调用。"""
        global sync_chunk_generation_counter
        
        # 如果是同步生成，减少计数器
        if sync_chunk_generation_counter > 0:
            sync_chunk_generation_counter -= 1
            
        if not blocks_data:
            self.is_generated = True # 标记为已生成，即使没有方块
            return

        # 缓存生成的数据
        cache_key = f"chunk_{self.position[0]}_{self.position[1]}"
        if not hasattr(self.__class__, "_chunk_cache"):
            self.__class__._chunk_cache = {}

        # 限制缓存大小
        if len(self.__class__._chunk_cache) > 50:
            distances = {}
            player_chunk_pos = get_chunk_position(player.position) if player and hasattr(player, 'position') else self.position
            for key in list(self.__class__._chunk_cache.keys()): # 使用 list 避免迭代时修改
                try:
                    parts = key.split('_')
                    if len(parts) >= 3:
                        x = int(parts[1])
                        z = int(parts[2])
                        dist = abs(x - player_chunk_pos[0]) + abs(z - player_chunk_pos[1])
                        distances[key] = dist
                    else:
                        distances[key] = float('inf') # 无效key，优先删除
                except (ValueError, IndexError):
                    distances[key] = float('inf')

            if distances:
                key_to_remove = max(distances.items(), key=lambda item: item[1])[0]
                if key_to_remove in self.__class__._chunk_cache:
                     del self.__class__._chunk_cache[key_to_remove]

        self.__class__._chunk_cache[cache_key] = blocks_data

        # 调用创建方块逻辑，根据参数决定是否使用分批处理
        if use_batching:
            # 分批创建方块，避免一次创建过多对象
            self.create_blocks_batched(blocks_data)
        else:
            # 使用原有方法创建方块
            self.create_blocks(blocks_data)

        # 在所有创建逻辑之后标记为已生成
        self.is_generated = True

    def generate(self):
        global sync_chunk_generation_counter
        
        if self.is_generated or self._future:
            return
        
        # 检查是否有缓存数据
        cache_key = f"chunk_{self.position[0]}_{self.position[1]}"
        cached_data = getattr(self.__class__, "_chunk_cache", {}).get(cache_key)
        
        if cached_data:
            # 使用缓存数据 - 调用 finalize 方法处理
            self._finalize_chunk_creation(cached_data)
            return
        
        # 获取当前帧率，用于性能保护
        current_fps = getattr(application, 'fps', 30)
        
        # 检查是否是玩家下方区块 - 如果是，考虑使用同步生成
        is_below_player = False
        if player and hasattr(player, 'position'):
            # 计算区块中心位置
            chunk_center_x = self.position[0] * CHUNK_SIZE + CHUNK_SIZE/2
            chunk_center_z = self.position[1] * CHUNK_SIZE + CHUNK_SIZE/2
            
            # 检查是否在玩家正下方或附近
            player_x, player_y, player_z = player.position
            # 缩小下方区块的检测范围，只对最关键的区块使用同步生成
            if ((abs(player_x - chunk_center_x) < CHUNK_SIZE and 
                abs(player_z - chunk_center_z) < CHUNK_SIZE and 
                player_y - 15 < 0)):  # 只检查玩家正下方15格内的区块
                is_below_player = True
        
        # 对于玩家下方区块，在满足条件时使用同步生成
        if is_below_player and sync_chunk_generation_counter < MAX_SYNC_CHUNKS_PER_FRAME and current_fps > MIN_FPS_FOR_SYNC_GENERATION:
            try:
                # 增加同步生成计数器
                sync_chunk_generation_counter += 1
                
                # 直接在主线程生成地形数据
                blocks_data = self.generate_terrain_data()
                
                # 缓存生成的数据
                if not hasattr(self.__class__, "_chunk_cache"):
                    self.__class__._chunk_cache = {}
                
                # 限制缓存大小
                if len(self.__class__._chunk_cache) > 50:
                    # 删除最远的区块缓存
                    distances = {}
                    for key in self.__class__._chunk_cache.keys():
                        try:
                            parts = key.split('_')
                            if len(parts) >= 3:
                                x = int(parts[1])
                                z = int(parts[2])
                                dist = abs(x - self.position[0]) + abs(z - self.position[1])
                                distances[key] = dist
                        except (ValueError, IndexError):
                            distances[key] = 0
                    
                    if distances:
                        key_to_remove = max(distances.items(), key=lambda x: x[1])[0]
                        del self.__class__._chunk_cache[key_to_remove]
                
                # 保存到缓存
                self.__class__._chunk_cache[cache_key] = blocks_data
                
                # 立即处理区块创建，但使用分批处理避免一次创建过多方块
                self._finalize_chunk_creation(blocks_data, use_batching=True)
                return
            except Exception as e:
                print(f"Error in synchronous chunk generation at {self.position}: {e}")
                # 减少同步生成计数器
                sync_chunk_generation_counter -= 1
                # 如果同步生成失败，回退到异步生成
        
        # 对于非紧急区块或当帧率过低时，使用异步生成
        def generate_task():
            try:
                # 生成地形数据
                blocks_data = self.generate_terrain_data()
                
                # 缓存生成的数据
                if not hasattr(self.__class__, "_chunk_cache"):
                    self.__class__._chunk_cache = {}
                
                # 限制缓存大小，最多保存50个区块的数据
                if len(self.__class__._chunk_cache) > 50:
                    # 删除最远的区块缓存，而不是随机删除
                    # 计算所有缓存区块与当前区块的距离
                    distances = {}
                    for key in self.__class__._chunk_cache.keys():
                        # 从键中提取坐标
                        try:
                            parts = key.split('_')
                            if len(parts) >= 3:
                                x = int(parts[1])
                                z = int(parts[2])
                                # 计算曼哈顿距离
                                dist = abs(x - self.position[0]) + abs(z - self.position[1])
                                distances[key] = dist
                        except (ValueError, IndexError):
                            distances[key] = 0  # 解析失败时默认距离为0
                    
                    # 找出距离最远的区块
                    if distances:
                        key_to_remove = max(distances.items(), key=lambda x: x[1])[0]
                        del self.__class__._chunk_cache[key_to_remove]
                
                # 注意：缓存移到 _finalize_chunk_creation 中处理
                return blocks_data
            except Exception as e:
                logging.error(f"Error generating chunk data at {self.position}: {e}", exc_info=True)
                return []  # 返回空列表避免错误传播
        
        def on_done(future):
            try:
                blocks_data = future.result()
                # 使用 invoke 确保在主线程调用 _finalize_chunk_creation
                # 移除延迟，尽快处理
                invoke(lambda: self._finalize_chunk_creation(blocks_data, use_batching=True), delay=0)
            except Exception as e:
                logging.error(f"Error processing chunk future at {self.position}: {e}", exc_info=True)
                # 即使出错也标记为已生成，防止无限重试
                # finalize 内部会处理 is_generated
                # 如果 finalize 本身失败，这里需要确保标记
                if not self.is_generated:
                    try:
                        # 尝试最后一次标记，即使 finalize 失败
                        self.is_generated = True
                        logging.warning(f"Chunk {self.position} marked as generated after future processing error.")
                    except Exception as final_err:
                        logging.error(f"Failed to mark chunk {self.position} as generated after error: {final_err}")
        
        # 提交任务到线程池
        self._future = chunk_executor.submit(generate_task)
        self._future.add_done_callback(on_done)
        
    def create_blocks_batched(self, blocks_data):
        """使用更激进的分批处理创建方块，避免主线程阻塞"""
        # 检查是否是玩家下方的关键区块
        is_below_player = False
        if player and hasattr(player, 'position'):
            # 计算区块中心位置
            chunk_center_x = self.position[0] * CHUNK_SIZE + CHUNK_SIZE/2
            chunk_center_z = self.position[1] * CHUNK_SIZE + CHUNK_SIZE/2
            
            # 检查是否在玩家正下方
            player_x, player_y, player_z = player.position
            if (abs(player_x - chunk_center_x) < CHUNK_SIZE and 
                abs(player_z - chunk_center_z) < CHUNK_SIZE and 
                player_y - 10 < 0):  # 玩家下方10格内的区块
                is_below_player = True
        
        # 分离顶部方块和其他方块
        top_blocks = []
        other_blocks = []
        
        # 找出每个x,z坐标的最高方块
        highest_blocks = {}
        for pos, block_id in blocks_data:
            key = (pos.x, pos.z)
            if key not in highest_blocks or pos.y > highest_blocks[key][0].y:
                highest_blocks[key] = (pos, block_id)
        
        # 将最高方块添加到top_blocks，其余添加到other_blocks
        for pos, block_id in blocks_data:
            key = (pos.x, pos.z)
            if (pos, block_id) == highest_blocks[key]:
                top_blocks.append((pos, block_id))
            else:
                other_blocks.append((pos, block_id))
        
        # 对于玩家下方区块，立即创建顶部方块，其余分批创建
        if is_below_player:
            # 立即创建顶部方块
            if top_blocks:
                self._create_batch(top_blocks)
            
            # 分批创建其他方块，使用更小的批次和更长的延迟
            batch_size = 20  # 减小批次大小
            for i in range(0, len(other_blocks), batch_size):
                batch_data = other_blocks[i:i+batch_size]
                # 使用更长的延迟，减轻主线程负担
                delay = 0.05 + (i / len(other_blocks)) * 0.2
                invoke(lambda b=batch_data: self._create_batch(b), delay=delay)
        else:
            # 非关键区块，所有方块都分批创建
            all_blocks = top_blocks + other_blocks
            batch_size = 30
            for i in range(0, len(all_blocks), batch_size):
                batch_data = all_blocks[i:i+batch_size]
                delay = 0.1 + (i / len(all_blocks)) * 0.3
                invoke(lambda b=batch_data: self._create_batch(b), delay=delay)
    
    def create_blocks(self, blocks_data):
        # 优化方块创建过程，减少锁竞争和内存分配
        # 检查是否是玩家下方的关键区块
        is_below_player = False
        if player and hasattr(player, 'position'):
            # 计算区块中心位置
            chunk_center_x = self.position[0] * CHUNK_SIZE + CHUNK_SIZE/2
            chunk_center_z = self.position[1] * CHUNK_SIZE + CHUNK_SIZE/2
            
            # 检查是否在玩家正下方或附近
            player_x, player_y, player_z = player.position
            # 扩大判断范围，确保玩家下方和周围的区块都能快速生成
            if ((abs(player_x - chunk_center_x) < CHUNK_SIZE * 1.5 and 
                abs(player_z - chunk_center_z) < CHUNK_SIZE * 1.5 and 
                player_y - 15 > 0) or  # 玩家上方的区块
                (abs(player_x - chunk_center_x) < CHUNK_SIZE * 2 and 
                abs(player_z - chunk_center_z) < CHUNK_SIZE * 2 and 
                player_y - 30 < 0)):  # 玩家下方30格内的区块
                is_below_player = True
        
        # 对于玩家下方的关键区块，使用更快的创建方式
        if is_below_player:
            # 优先创建顶部方块，确保玩家有落脚点
            top_blocks = []
            near_blocks = []  # 靠近顶部的方块
            other_blocks = []
            
            # 分离顶部方块和其他方块
            for pos, block_id in blocks_data:
                # 找出每个x,z坐标的最高方块
                is_top = True
                is_near_top = False
                highest_y = -999
                
                # 找出每个x,z坐标的最高点
                for other_pos, _ in blocks_data:
                    if pos.x == other_pos.x and pos.z == other_pos.z:
                        highest_y = max(highest_y, other_pos.y)
                        if other_pos.y > pos.y:
                            is_top = False
                
                # 判断是否靠近顶部
                if not is_top and highest_y - pos.y <= 3:  # 距离顶部3格以内的方块
                    is_near_top = True
                
                if is_top:
                    top_blocks.append((pos, block_id))
                elif is_near_top:
                    near_blocks.append((pos, block_id))
                else:
                    other_blocks.append((pos, block_id))
            
            # 立即创建顶部方块
            if top_blocks:
                self._create_batch(top_blocks)
            
            # 立即创建靠近顶部的方块
            if near_blocks:
                self._create_batch(near_blocks)
            
            # 然后创建其他方块
            batch_size = 40  # 进一步增加批次大小，加快创建速度
            for i in range(0, len(other_blocks), batch_size):
                batch_data = other_blocks[i:i+batch_size]
                # 使用更短的延迟
                delay = (i / len(other_blocks)) * 0.02  # 进一步减少延迟时间
                invoke(lambda b=batch_data: self._create_batch(b), delay=delay)
        else:
            # 非关键区块使用标准创建过程
            batch_size = 25  # 增加批次大小，从20增加到25
            total_blocks = len(blocks_data)
            
            # 如果数据量很大，分散创建过程
            if total_blocks > 100:
                # 将大量方块的创建分散到多个帧
                for start_idx in range(0, total_blocks, batch_size):
                    end_idx = min(start_idx + batch_size, total_blocks)
                    # 使用延迟调用，但减少延迟时间
                    delay = (start_idx / batch_size) * 0.04  # 从0.05减少到0.04
                    batch_data = blocks_data[start_idx:end_idx]
                    invoke(lambda b=batch_data: self._create_batch(b), delay=delay)
            else:
                # 数据量较小时直接创建
                for i in range(0, total_blocks, batch_size):
                    batch_data = blocks_data[i:i+batch_size]
                    self._create_batch(batch_data)
        
        # is_generated 标志现在由 _finalize_chunk_creation 处理
        pass
    
    def _create_batch(self, batch_data):
        # 创建一批方块，大幅优化对象池使用
        # 减少锁的持有时间，只在必要时加锁
        blocks_to_add = []
        blocks_added_to_grid = [] # 记录添加到网格的方块
        
        # 获取当前区块已有的方块位置，用于快速查找
        existing_block_positions = {str(b.position) for b in self.blocks}

        for pos, block_id in batch_data:
            # 检查该位置是否已经有方块
            if str(pos) in existing_block_positions:
                continue # 如果已存在方块，则跳过

            # 尝试从对象池获取方块
            block = None
            # 大幅提高对象池使用率，从30%提高到80%
            if Chunk._block_pool:
                try:
                    # 使用线程安全的方式从对象池获取方块
                    with chunk_lock:
                        if Chunk._block_pool:  # 再次检查，确保池不为空
                            block = Chunk._block_pool.pop()
                    
                    # 重置方块属性 - 在锁外执行以减少锁持有时间
                    if block:
                        block.position = pos
                        block.id = block_id
                        block.texture = Block_list[block_id]
                        block.visible = True
                        # 重置其他属性
                        block.collision_cooldown = 0
                        block.last_update = 0
                        block.last_distance_check = 0
                        block.in_range = False
                        block.distance_to_player = 999
                except Exception:
                    block = None  # 如果重用失败，创建新方块
            
            # 如果没有从对象池获取到方块，创建新方块
            if block is None:
                block = Block(position=pos, id=block_id)
                # 水方块相关代码已移除
            
            blocks_to_add.append(block)
            blocks_added_to_grid.append(block) # 同时记录到待添加网格列表
            existing_block_positions.add(str(pos)) # 将新添加的方块位置加入集合
        
        # 批量添加方块到区块和空间网格，减少锁竞争
        if blocks_to_add: # 只有当有新的方块需要添加时才执行
            with chunk_lock:
                self.blocks.extend(blocks_to_add)
            # 在锁外批量添加到空间网格
            for block_to_add_to_grid in blocks_added_to_grid: # 使用不同的变量名以避免混淆
                spatial_grid.add_block(block_to_add_to_grid)
    
    def generate_terrain_data(self):
        # 优化地形生成算法
        blocks_data = []
        chunk_x, chunk_z = self.position
        
        # 预计算所有高度值
        heights = {}
        # 湖泊相关代码已移除
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                wx = x + chunk_x * CHUNK_SIZE
                wz = z + chunk_z * CHUNK_SIZE
                # 地形高度
                heights[(x, z)] = floor(noise([wx/scale, wz/scale]) * height_scale + base_height)
        
        # 批量生成地形方块
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                wx = x + chunk_x * CHUNK_SIZE
                wz = z + chunk_z * CHUNK_SIZE
                terrain_height = heights[(x, z)]
                # 湖泊相关代码已移除

                # 生成地形方块
                # 顶层草方块
                blocks_data.append((Vec3(wx, terrain_height, wz), GRASS))
                
                # 泥土层（顶层-1到顶层-3）
                dirt_bottom = max(terrain_height-3, -4) # 确保不会生成到基岩之下
                for y_dirt in range(terrain_height-1, dirt_bottom, -1):
                    blocks_data.append((Vec3(wx, y_dirt, wz), DIRT))
                
                # 石头层（紧接着泥土层，一直到底部）
                stone_top = dirt_bottom -1
                stone_bottom = -10  # 石头层底部深度
                if stone_top > stone_bottom:
                    for y_stone in range(stone_top, stone_bottom, -1):
                        blocks_data.append((Vec3(wx, y_stone, wz), STONE))
                
                # 最底层基岩
                blocks_data.append((Vec3(wx, -3, wz), BED)) # 使用 BED 作为基岩ID，假设它是坚不可摧的

                # 湖泊生成代码已移除

        # 生成洞穴系统
        cave_noise_3d = PerlinNoise(octaves=2, seed=int(time.time()) + 2)  # 3D噪声用于洞穴
        CAVE_THRESHOLD = 0.6  # 洞穴阈值，可调整
        CAVE_SCALE = 30  # 洞穴噪声缩放

        temp_blocks_data = []
        existing_block_positions = {bd[0]: bd[1] for bd in blocks_data}

        for x_offset in range(CHUNK_SIZE):
            for z_offset in range(CHUNK_SIZE):
                wx = x_offset + chunk_x * CHUNK_SIZE
                wz = z_offset + chunk_z * CHUNK_SIZE
                # 洞穴主要在石头层生成，避免影响地表和基岩
                for y_offset in range(heights[(x_offset, z_offset)] - 4, -2): # 从石头层顶部向下到基岩上方
                    # 获取3D噪声值
                    noise_val = cave_noise_3d([wx / CAVE_SCALE, y_offset / CAVE_SCALE, wz / CAVE_SCALE])
                    current_pos = Vec3(wx, y_offset, wz)
                    # 如果噪声值超过阈值，并且当前位置是石头，则移除石头（形成洞穴）
                    if noise_val > CAVE_THRESHOLD and existing_block_positions.get(current_pos) == STONE:
                        # 从 blocks_data 中移除该石头方块
                        blocks_data = [bd for bd in blocks_data if not (bd[0] == current_pos and bd[1] == STONE)]
                        existing_block_positions.pop(current_pos, None) # 更新记录
        
        # 优化树木生成 - 极大减少树木密度并简化生成逻辑
        # 每个区块最多生成1棵树，但概率很低，避免树木过于密集
        max_trees = 1
        tree_count = 0
        
        # 随机选择位置生成树木，而不是遍历每个方块
        tree_candidates = []
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                # 树木只在草地上生成，并且不在水下
                current_height = heights[(x,z)]
                # 水位检查代码已移除
                # 确保树木不会生成在洞穴入口
                is_cave_entrance = False
                if not any(bd[0] == Vec3(x + chunk_x * CHUNK_SIZE, current_height, z + chunk_z * CHUNK_SIZE) for bd in blocks_data):
                    is_cave_entrance = True

                if not is_cave_entrance and current_height > base_height and random.random() < 0.01:  # 降低单个位置生成树的概率到1%
                    tree_candidates.append((x, z, current_height))
        
        # 随机选择最多max_trees个位置生成树
        if tree_candidates:
            # 随机打乱候选位置 - 使用已导入的函数而不是局部导入
            from random import shuffle
            shuffle(tree_candidates)
            
            # 选择前max_trees个位置生成树
            for x, z, height in tree_candidates[:max_trees]:
                wx = x + chunk_x * CHUNK_SIZE
                wz = z + chunk_z * CHUNK_SIZE
                self.add_tree(blocks_data, wx, wz, height)
                tree_count += 1
                if tree_count >= max_trees:
                    break
        
        return blocks_data
    
    def add_tree(self, blocks_data, wx, wz, base_height):
        tree_height = randint(4, 6)
        
        # 树干
        blocks_data.extend(
            (Vec3(wx, base_height + 1 + i, wz), 4)
            for i in range(tree_height)
        )
        
        # 树叶金字塔形状生成
        top_y = base_height + tree_height + 1
        
        # 金字塔形状的树叶 - 从上到下分层生成
        # 顶层 - 最小的一层
        blocks_data.append((Vec3(wx, top_y, wz), 5))
        
        # 第二层 - 3x3 正方形
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                blocks_data.append((Vec3(wx + dx, top_y - 1, wz + dz), 5))
        
        # 第三层 - 5x5 正方形
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                # 边缘稀疏处理，让树叶看起来更自然
                if abs(dx) == 2 and abs(dz) == 2:
                    if random.random() < 0.5:  # 只有50%的概率生成角落的树叶
                        continue
                blocks_data.append((Vec3(wx + dx, top_y - 2, wz + dz), 5))
        
        # 第四层 - 7x7 正方形，底部最大的一层
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                # 边缘稀疏处理
                if abs(dx) == 3 and abs(dz) == 3:
                    continue  # 不在最远的角落生成树叶
                if (abs(dx) == 3 or abs(dz) == 3) and random.random() < 0.3:
                    continue  # 边缘有30%的概率不生成树叶，使外观更自然
                blocks_data.append((Vec3(wx + dx, top_y - 3, wz + dz), 5))

        

    def destroy(self):
        try:
            # 大幅优化销毁过程，提高对象池使用率
            # 批量处理方块，将更多方块放入对象池而不是直接销毁
            blocks_to_process = []
            for block in self.blocks:
                if isinstance(block, Block):
                    blocks_to_process.append(block)
            
            # 进一步增加批次大小，加快处理速度
            batch_size = 15  # 从10增加到15
            
            # 计算要放入对象池的方块数量上限
            # 确保对象池不会过大，但也不会太小
            max_pool_blocks = min(len(blocks_to_process), Chunk._block_pool.maxlen - len(Chunk._block_pool))
            # 确保至少有一些方块会被放入对象池
            pool_blocks_count = max(max_pool_blocks // 2, 1)
            
            # 优先处理要放入对象池的方块
            pool_candidates = blocks_to_process[:pool_blocks_count]
            destroy_candidates = blocks_to_process[pool_blocks_count:]
            
            # 处理要放入对象池的方块
            for i in range(0, len(pool_candidates), batch_size):
                batch = pool_candidates[i:i+batch_size]
                # 使用延迟调用，分散处理压力
                invoke(lambda b=batch: self._process_pool_batch(b), delay=i*0.02)
            
            # 处理要销毁的方块
            for i in range(0, len(destroy_candidates), batch_size):
                batch = destroy_candidates[i:i+batch_size]
                # 使用延迟调用，分散处理压力
                invoke(lambda b=batch: self._process_destroy_batch(b), delay=i*0.02)
            
            # 清理引用
            self.blocks.clear()
            self.is_generated = False
            
            # 取消任何正在进行的异步任务
            if self._future and not self._future.done():
                try:
                    self._future.cancel()
                except Exception:
                    pass  # 忽略取消异常
            self._future = None
            
            # 只在必要时进行垃圾回收
            # 使用静态计数器控制GC频率
            if not hasattr(Chunk.destroy, 'gc_counter'):
                Chunk.destroy.gc_counter = 0
            
            Chunk.destroy.gc_counter += 1
            # 每销毁30个区块才进行一次垃圾回收，进一步减少GC频率
            if Chunk.destroy.gc_counter >= 30:  # 从20增加到30
                Chunk.destroy.gc_counter = 0
                # 使用非阻塞方式触发垃圾回收
                invoke(gc.collect, delay=0.5)  # 增加延迟，进一步降低卡顿
        except Exception as e:
            # 确保即使出错也清理引用，避免内存泄漏
            self.blocks.clear()
            self.is_generated = False
            self._future = None
    
    def _process_pool_batch(self, batch):
        """处理要放入对象池的方块批次"""
        with chunk_lock:  # 使用锁保护对象池
            for block in batch:
                try:
                    # 隐藏方块但不销毁
                    block.visible = False
                    # 将方块添加到对象池
                    if len(Chunk._block_pool) < Chunk._block_pool.maxlen:
                        Chunk._block_pool.append(block)
                    else:
                        # 对象池已满，销毁方块
                        destroy(block, delay=0.01)
                except Exception:
                    # 出错时直接销毁
                    try:
                        destroy(block)
                    except:
                        pass
    
    def _process_destroy_batch(self, batch):
        """处理要销毁的方块批次"""
        for block in batch:
            try:
                # 添加更长的延迟，进一步分散销毁压力
                destroy(block, delay=uniform(0.001, 0.01))  # 减少销毁延迟提升响应速度
            except Exception:
                pass  # 忽略错误

def get_chunk_position(position):
    """根据世界坐标获取区块坐标"""
    # 使用地板除法确保负数坐标也能正确工作
    x = floor(position.x / CHUNK_SIZE)
    z = floor(position.z / CHUNK_SIZE)
    return (x, z)

def new_chunk(chunk_pos):
    """异步加载单个区块"""
    try:
        # 检查区块是否已经加载，避免重复加载
        if chunk_pos not in loaded_chunks:
            # 创建并生成区块
            chunk = Chunk(chunk_pos)
            # generate 方法内部会处理异步或同步生成
            chunk.generate() 
            # 将区块添加到 loaded_chunks 的操作现在由 generate 的回调或同步逻辑处理
            # loaded_chunks[chunk_pos] = chunk # 不再在这里添加
            # 减少日志输出
            # print(f"Chunk {chunk_pos} generation initiated.")
    except Exception as e:
        # 只在出错时输出日志
        print(f"Error in new_chunk: {e}")
        # 出错时不要阻塞，继续执行
executor = ThreadPoolExecutor(max_workers=4)  # 设置线程池大小

# 添加生成树的函数
def generate_tree_in_chunk(chunk_pos):
    """在指定区块中生成树"""
    if chunk_pos not in loaded_chunks:
        return
        
    chunk = loaded_chunks[chunk_pos]
    
    # 找到合适的位置生成树
    grass_blocks = []
    for block in chunk.blocks:
        if hasattr(block, 'id') and block.id == 0:  # 草方块
            # 检查上方是否有空间
            has_space = True
            for other_block in chunk.blocks:
                if (hasattr(other_block, 'position') and 
                    other_block.position.x == block.position.x and
                    other_block.position.z == block.position.z and
                    other_block.position.y > block.position.y):
                    has_space = False
                    break
            if has_space:
                grass_blocks.append(block)
    
    if not grass_blocks:
        return
    
    # 随机选择一个草方块作为树的基座
    base_block = choice(grass_blocks)
    base_pos = base_block.position
    
    # 生成树干(3-5格高)
    tree_height = randint(3, 5)
    for y in range(1, tree_height + 1):
        log_pos = Vec3(base_pos.x, base_pos.y + y, base_pos.z)
        # 检查位置是否已有方块
        position_occupied = False
        for block in chunk.blocks:
            if hasattr(block, 'position') and block.position == log_pos:
                position_occupied = True
                break
        
        if not position_occupied:
            log_block = Block(position=log_pos, id=4)  # 原木方块
            chunk.blocks.append(log_block)
    
    # 生成树叶(在树干顶部周围)
    for dx in range(-2, 3):
        for dy in range(-1, 2):
            for dz in range(-2, 3):
                # 跳过树干位置
                if dx == 0 and dz == 0 and dy < 1:
                    continue
                    
                # 计算到树干中心的距离
                distance = abs(dx) + abs(dy) + abs(dz)
                # 距离越远，生成叶子的概率越低
                if distance <= 3 and random.random() < (1.0 - distance/4):
                    leaf_pos = Vec3(
                        base_pos.x + dx,
                        base_pos.y + tree_height + dy,
                        base_pos.z + dz
                    )
                    
                    # 检查位置是否已有方块
                    position_occupied = False
                    for block in chunk.blocks:
                        if hasattr(block, 'position') and block.position == leaf_pos:
                            position_occupied = True
                            break
                    
                    if not position_occupied:
                        leaf_block = Block(position=leaf_pos, id=5)  # 树叶方块
                        chunk.blocks.append(leaf_block)

def update_chunks():
    # 如果玩家不存在，直接返回
    if not hasattr(player, 'position'):
        return
    
    # 获取当前帧率，用于性能保护
    current_fps = getattr(application, 'fps', 30)
    
    # 获取玩家所在区块
    player_chunk = get_chunk_position(player.position)
    
    # 使用螺旋序列加载周围区块
    spiral_chunks = get_spiral_sequence(player_chunk, RENDER_DISTANCE)
    
    # 性能保护 - 在帧率低时减少区块生成
    max_chunks_to_load = 1  # 默认每次最多加载1个新区块
    if current_fps < 15:  # 帧率低于15时减少加载
        # 只有1/3的概率加载新区块
        if random.random() > 0.33:
            return
    elif current_fps > 30:  # 帧率高时可以加载更多
        max_chunks_to_load = 2
    
    # 跟踪本次更新已加载的区块数量
    chunks_loaded = 0
    
    # 优先加载玩家下方区块
    below_pos = get_chunk_position(Vec3(player.position.x, player.position.y - 10, player.position.z))
    if below_pos not in loaded_chunks and chunks_loaded < max_chunks_to_load:
        # 创建并生成区块
        chunk = Chunk(below_pos)
        chunk.generate()
        loaded_chunks[below_pos] = chunk
        chunks_loaded += 1
    
    # 然后加载其他区块
    for chunk_pos in spiral_chunks:
        if chunk_pos not in loaded_chunks and chunks_loaded < max_chunks_to_load:
            # 创建并生成区块
            chunk = Chunk(chunk_pos)
            chunk.generate()
            loaded_chunks[chunk_pos] = chunk
            chunks_loaded += 1
            if chunks_loaded >= max_chunks_to_load:
                break
    
    # 使用静态变量控制更新频率
    current_time = time.time()
    if not hasattr(update_chunks, 'last_update'):
        update_chunks.last_update = 0
    
    # 大幅减少更新间隔，确保玩家下方区块能立即生成
    # 将间隔从UPDATE_INTERVAL减少到0.05秒，确保快速响应
    if current_time - update_chunks.last_update < 0.05:
        return
    
    update_chunks.last_update = current_time
    
    # 使用静态变量控制性能监控频率
    if not hasattr(update_chunks, 'perf_counter'):
        update_chunks.perf_counter = 0
    
    update_chunks.perf_counter += 1
    measure_perf = update_chunks.perf_counter % 10 == 0  # 每10次测量一次性能
    
    # 只在需要时记录开始时间
    start_time = time.time() if measure_perf else 0
    
    try:
        player_chunk = get_chunk_position(player.position)
        
        # 使用静态变量记录上次玩家位置
        if not hasattr(update_chunks, 'last_player_chunk'):
            update_chunks.last_player_chunk = player_chunk
        
        # 检测玩家是否移动到新区块
        player_moved = update_chunks.last_player_chunk != player_chunk
        
        # 更新玩家位置记录
        update_chunks.last_player_chunk = player_chunk
        
        # 计算玩家朝向，优先加载玩家面向的区块
        if hasattr(player, 'rotation_y'):
            facing_x = int(round(sin(player.rotation_y * pi/180)))
            facing_z = int(round(cos(player.rotation_y * pi/180)))
            # 确保朝向向量不为零向量
            if facing_x == 0 and facing_z == 0:
                facing_x, facing_z = 0, 1  # 如果计算结果为零向量，使用默认朝向
        else:
            facing_x, facing_z = 0, 1  # 默认朝向
        
        # 检查玩家当前所在区块是否已生成，如果没有则立即生成
        # 这是防止玩家掉落的关键优化
        current_chunk = get_chunk_position(player.position)
        if current_chunk not in loaded_chunks or not loaded_chunks[current_chunk].is_generated:
                try:
                    # 立即生成玩家所在区块，优先级最高
                    chunk = Chunk(current_chunk)
                    # 强制同步生成，不使用异步
                    chunk.generate()
                    loaded_chunks[current_chunk] = chunk
                    # 强制立即生成，不延迟
                    chunk.is_generated = True
                except Exception as e:
                    print(f"Error generating current player chunk: {e}")

        
        # 使用优化的区块加载系统更新区块
        if not hasattr(update_chunks, 'load_counter'):
            update_chunks.load_counter = 0
        
        update_chunks.load_counter += 1
        # 增加更新频率，确保区块能及时生成
        should_update_chunks = player_moved or update_chunks.load_counter >= 3  # 从5减少到3
        
        if should_update_chunks:
            update_chunks.load_counter = 0
            
            # 使用区块加载优化器更新区块
            chunk_loading_optimizer.update(player.position, player.forward)
            
            # 兼容旧系统，保留部分代码
            chunks_to_load = []
            
            # 增加渲染距离，确保玩家不会掉落
            reduced_distance = max(3, RENDER_DISTANCE)  # 从2增加到3
            
            # 获取玩家下方所有区块位置
            below_chunks = []
            for depth in range(1, 20, 5):  # 检查更多下方区块
                below_pos = get_chunk_position(Vec3(player.position.x, player.position.y - depth, player.position.z))
                below_chunks.append(below_pos)
            
            # 检查需要加载的区块
            for x in range(player_chunk[0] - reduced_distance, player_chunk[0] + reduced_distance + 1):
                for z in range(player_chunk[1] - reduced_distance, player_chunk[1] + reduced_distance + 1):
                    if (x, z) not in loaded_chunks:
                        # 计算曼哈顿距离
                        dist = abs(x - player_chunk[0]) + abs(z - player_chunk[1])
                        
                        # 计算垂直位置权重 - 玩家下方区块优先级最高
                        vertical_weight = 1
                        # 检查这个区块是否在玩家下方
                        if (x, z) in below_chunks:
                            # 根据深度调整权重，越近的下方区块权重越高
                            depth_index = below_chunks.index((x, z))
                            vertical_weight = 20 - depth_index * 3  # 20, 17, 14...
                        
                        # 计算朝向权重 - 玩家面向的区块优先级更高
                        direction_weight = 1
                        dx, dz = x - player_chunk[0], z - player_chunk[1]
                        if dx * facing_x + dz * facing_z > 0:  # 如果在玩家前方
                            direction_weight = 2
                        
                        # 计算移动预测权重 - 根据玩家移动方向预测
                        movement_weight = 1
                        if hasattr(update_chunks, 'prev_player_chunk'):
                            prev_chunk = update_chunks.prev_player_chunk
                            # 计算移动方向
                            move_dx = player_chunk[0] - prev_chunk[0]
                            move_dz = player_chunk[1] - prev_chunk[1]
                            # 如果区块在移动方向上
                            if move_dx != 0 or move_dz != 0:  # 确保玩家确实在移动
                                if (move_dx * dx > 0 or move_dz * dz > 0):
                                    movement_weight = 2.0  # 从1.5提高到2.0
                        
                        # 综合计算优先级 - 值越小优先级越高
                        denominator = vertical_weight * direction_weight * movement_weight
                        # 避免除零错误
                        if denominator == 0:
                            denominator = 0.001  # 使用一个小的非零值
                        priority = dist / denominator
                        
                        chunks_to_load.append(((x, z), priority))
            
            # 更新上一次玩家区块位置，用于移动预测
            update_chunks.prev_player_chunk = player_chunk
            
            # 优化排序 - 按优先级排序
            if chunks_to_load:
                chunks_to_load.sort(key=lambda x: x[1])
                # 增加每次加载的区块数量，确保关键区块能及时生成
                chunks_to_load = chunks_to_load[:MAX_CHUNKS_PER_UPDATE + 1]
            
            # 批量加载区块 - 优先加载最重要的区块
            if chunks_to_load:  # 确保有区块需要加载
                # 增加每次加载的区块数量
                max_chunks = min(3, len(chunks_to_load))  # 从2增加到3
                
                for i in range(max_chunks):
                    if chunks_to_load:  # 再次检查，确保列表不为空
                        chunk_pos, priority = chunks_to_load.pop(0)
                        # 直接在主线程中创建区块，避免线程池问题
                        if chunk_pos not in loaded_chunks:
                            try:
                                chunk = Chunk(chunk_pos)
                                # 对于高优先级区块（如玩家下方），强制立即生成
                                if priority < 0.5:  # 高优先级区块
                                    chunk.generate()
                                    chunk.is_generated = True  # 强制标记为已生成
                                else:
                                    chunk.generate()
                                loaded_chunks[chunk_pos] = chunk
                            except Exception as e:
                                print(f"Error generating chunk at {chunk_pos}: {e}")
                    else:
                        break  # 如果列表为空，提前退出循环
        
        # 优化区块卸载 - 使用缓冲区避免频繁加载/卸载
        # 只在玩家移动或每隔一段时间才执行区块卸载
        if not hasattr(update_chunks, 'unload_counter'):
            update_chunks.unload_counter = 0
        
        update_chunks.unload_counter += 1
        should_unload = player_moved or update_chunks.unload_counter >= 8
        
        if should_unload:
            update_chunks.unload_counter = 0
            
            # 只检查一部分区块，分散卸载压力
            chunk_keys = list(loaded_chunks.keys())
            if not chunk_keys:  # 避免空列表导致的索引错误
                return
                
            # 每次最多检查1个区块，减少卸载压力
            check_count = min(1, len(chunk_keys))
            
            # 使用静态计数器决定检查哪些区块
            if not hasattr(update_chunks, 'unload_index'):
                update_chunks.unload_index = 0
            
            for i in range(check_count):
                # 循环检查区块
                if len(chunk_keys) == 0:  # 再次检查，以防在循环中被修改
                    break
                    
                index = (update_chunks.unload_index + i) % len(chunk_keys)
                chunk_pos = chunk_keys[index]
                
                # 计算与玩家的距离
                dist = abs(chunk_pos[0] - player_chunk[0]) + abs(chunk_pos[1] - player_chunk[1])
                # 使用更大的缓冲区，减少频繁加载/卸载同一区块
                if dist > RENDER_DISTANCE + 4:
                    try:
                        if chunk_pos in loaded_chunks:  # 再次检查，以防在其他线程中被修改
                            loaded_chunks[chunk_pos].destroy()
                            del loaded_chunks[chunk_pos]
                    except Exception as e:
                        pass
            
            # 更新检查索引
            if chunk_keys and len(chunk_keys) > 0:  # 确保 chunk_keys 不为空且长度大于0
                update_chunks.unload_index = (update_chunks.unload_index + check_count) % max(1, len(chunk_keys))  # 使用 max 确保除数至少为1
    except Exception as e:
        print(f"Error in update_chunks: {e}")
        # 出错时重置状态，避免卡死
        # 出错时重置状态，避免卡死
        if hasattr(update_chunks, 'last_update'):
            update_chunks.last_update = 0
    finally:
        # 记录性能数据
        if measure_perf and start_time > 0:
            duration = time.time() - start_time
            if duration > 0.1:  # 只记录耗时较长的更新
                print(f"Chunk update took {duration:.2f} seconds")

# 修改初始化函数
def initialize_game():
    global hand, sky, fps_text, performance_stats_text, mesh_renderer  # 声明全局变量
    
    # 添加UI和其他组件
    sky = Sky(texture='assets/skybox.png')
    hand = Hand()
    
    # 初始化面片渲染器
    mesh_renderer = MeshSplittingRenderer()
    print("面片渲染器初始化完成")
    
    # 初始化区块加载优化器
    chunk_loading_optimizer.enabled = True
    # 预热区块缓存，提前加载玩家周围区块
    if player and hasattr(player, 'position'):
        preload_initial_chunks(player.position, distance=2)
        # 初始化渐进式加载系统
        initialize_on_game_start(player.position)
        print("渐进式加载系统初始化完成")
    
    # 创建性能统计显示
    create_performance_stats_display()
    
    # 添加调试信息
    print("游戏初始化完成，准备生成区块...")

# 修改 input 函数，添加性能优化快捷键
def input(key):
    global block_type_id, current_state, player, hand, sky, show_performance_stats, hotbar, chunk_loading_optimizer, ui_hidden

    # F1 键：隐藏/显示UI
    if key == 'f1':
        ui_hidden = not ui_hidden
        for entity in scene.entities:
            if hasattr(entity, 'is_ui') and entity.is_ui:
                entity.enabled = not ui_hidden
        if performance_stats_text:
            performance_stats_text.enabled = show_performance_stats and not ui_hidden
        print(f"UI {'隐藏' if ui_hidden else '显示'}")

    # F2 键：截图
    elif key == 'f2':
        base.screenshot(namePrefix='./screenshots/') # 保存到指定目录
        print("截图已保存！")
    elif key == 'f12':  # F12: 切换综合性能优化器
        enabled = comprehensive_optimizer.toggle()
        print(f"综合性能优化器: {'开启' if enabled else '关闭'}")
    # 只有在游戏状态下才处理这些输入
    if current_state == GameState.PLAYING:
        if key == 'escape':
            # 返回主菜单
            global main_menu
            current_state = GameState.MAIN_MENU
            
            # 销毁玩家和游戏元素
            if player:
                destroy(player)
                player = None
            
            # 销毁手和天空
            if 'hand' in globals() and hand:
                destroy(hand)
            if 'sky' in globals() and sky:
                destroy(sky)

            
            # 销毁物品栏
            if 'hotbar' in globals() and hotbar:
                destroy(hotbar.background)
                destroy(hotbar.selection_highlight)
                for icon in hotbar.item_icons:
                    destroy(icon)
                hotbar = None
            
            # 销毁所有已加载的区块
            for chunk_pos in list(loaded_chunks.keys()):
                loaded_chunks[chunk_pos].destroy()
                del loaded_chunks[chunk_pos]
            
            # 清理鼠标
            mouse.locked = False
            
            # 创建新的主菜单
            main_menu = create_main_menu()
            
        # 使用物品栏处理方块选择
        if hotbar and (key.isdigit() or key == 'scroll up' or key == 'scroll down'):
            if hotbar.input(key):  # 如果物品栏处理了输入
                block_type_id = hotbar.get_selected_block_id()
                return  # 物品栏已处理输入，不需要继续处理
        if key == 'shift':
            player.is_sneaking = True
            player.speed = 2  # 潜行速度
            player.jump_height = 0  # 禁用跳跃
        elif key == 'shift up':
            player.is_sneaking = False
            player.speed = 5  # 正常速度
            player.jump_height = 2  # 启用跳跃

# 添加插值函数
def lerp(start, end, t):
    """线性插值"""
    return start + (end - start) * t

class Hand(Entity):
    def __init__(self):
        super().__init__(
            parent=camera.ui,
            model='assets/arm.obj',
            texture=arm_texture,
            scale=0.2,
            rotation=Vec3(150, -10, 0),
            position=Vec2(100.6, -0.6)
        )
        self.is_ui = True

    def active(self):
        self.position = Vec2(0.6, -0.5)

    def pass_tive(self):
        self.position = Vec2(0.7, -0.6)

 # 猪类


# 修改生成初始区块的函数
def generate_initial_chunks():
    if not hasattr(player, 'position'):
        return
        
    player_chunk = get_chunk_position(player.position)
    print(f"生成初始区块，玩家位置：{player.position}，区块坐标：{player_chunk}")
    
    # 使用螺旋序列进行加载
    # 优先加载核心区块(3x3)
    for dx in range(-1, 2):
        for dz in range(-1, 2):
            chunk_pos = (player_chunk[0] + dx, player_chunk[1] + dz)
            if chunk_pos not in loaded_chunks:
                print(f"生成核心区块：{chunk_pos}")
                chunk = Chunk(chunk_pos)
                chunk.generate()
                loaded_chunks[chunk_pos] = chunk
    
    # 加载更多区块
    chunks_to_load = 5  # 每次加载的区块数量
    spiral_chunks = get_spiral_sequence(player_chunk, RENDER_DISTANCE)
    
    # 跳过已加载的核心区块
    for chunk_pos in spiral_chunks:
        if chunk_pos not in loaded_chunks:
            if chunk_pos is not None and chunk_pos not in loaded_chunks:
                print(f"生成螺旋区块：{chunk_pos}")
                chunk = Chunk(chunk_pos)
                chunk.generate()
                loaded_chunks[chunk_pos] = chunk
                chunk_loader.mark_chunk_loaded(chunk_pos)
    else:
        # 兼容旧代码，直接生成3x3区域
        # 首先生成玩家所在区块
        if player_chunk not in loaded_chunks:
            chunk = Chunk(player_chunk)
            chunk.generate()
            loaded_chunks[player_chunk] = chunk
        
        # 计算玩家朝向
        if hasattr(player, 'rotation_y'):
            facing_x = int(round(sin(player.rotation_y * pi/180)))
            facing_z = int(round(cos(player.rotation_y * pi/180)))
        else:
            facing_x, facing_z = 0, 1  # 默认朝向
        
        # 生成玩家周围的区块（3x3区域）
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                # 跳过已经生成的玩家所在区块
                if dx == 0 and dz == 0:
                    continue
                    
                chunk_pos = (player_chunk[0] + dx, player_chunk[1] + dz)
                if chunk_pos not in loaded_chunks:
                    print(f"生成周围区块：{chunk_pos}")
                    chunk = Chunk(chunk_pos)
                    chunk.generate()
                    loaded_chunks[chunk_pos] = chunk

# 主菜单按钮类
class MenuButton(Button):
    def __init__(self, text='', position=(0,0), on_click=None):
        super().__init__(
            parent=camera.ui,
            model='quad',
            text=text,
            position=position,
            color=color.gray.tint(-.2),
            highlight_color=color.gray.tint(-.1),
            pressed_color=color.gray.tint(-.3),
            scale=(0.8, 0.07),
            text_color=color.white,
            text_size=2
        )
        if on_click:
            self.on_click = on_click

# 主菜单类
class MainMenu(Entity):
    def __init__(self):
        super().__init__(parent=camera.ui)
        # 添加背景图片
        self.background = Entity(
            parent=self,
            model='quad',
            texture=load_texture('assets/background_blurred.png'),
            scale=(2.5, 1)
        )
        
        self.logo = Entity(
            parent=self,
            model='quad',
            texture=load_texture('assets/Java_Edition_Logo.png'),
            scale=(1, 0.19),
            position=(0, 0.25)
        )
        
        # 添加按钮
        button_y = 0
        self.singleplayer_button = MenuButton(
            text='单人游戏',
            position=(0, button_y),
            on_click=self.start_singleplayer
        )
        
        button_y -= 0.1
        self.multiplayer_button = MenuButton(
            text='多人游戏',
            position=(0, button_y),
            on_click=self.multiplayer
        )
        
        button_y -= 0.1
        self.realms_button = MenuButton(
            text='Minecraft Realms',
            position=(0, button_y),
            on_click=self.realms
        )
        
        button_y -= 0.1
        self.options_button = MenuButton(
            text='选项...',
            position=(0, button_y),
            on_click=self.options
        )
        
        button_y -= 0.1
        self.quit_button = MenuButton(
            text='退出游戏',
            position=(0, -0.4),
            on_click=self.quit_game
        )
    
    def start_singleplayer(self):
        global current_state, player, hand, sky, chunk_loader, main_menu
        
        # 销毁主菜单按钮和背景
        destroy(self.logo)
        destroy(self.singleplayer_button)
        destroy(self.multiplayer_button)
        destroy(self.realms_button)
        destroy(self.options_button)
        destroy(self.quit_button)
        
        # 确保销毁主菜单背景
        if hasattr(self, 'background') and self.background:
            destroy(self.background)
            self.background = None
        
        # 销毁主菜单对象本身
        destroy(self)
        main_menu = None
        
        # 直接初始化游戏，不使用加载界面
        def start_game_directly():
            # 初始化游戏
            initialize_game()



            # 创建玩家
            global player, hotbar
            player = FirstPersonController(
                # 恢复使用默认碰撞器
                jump_height=1.5,  # 降低跳跃高度，减少弹跳问题
                jump_duration=0.35,  # 减少跳跃持续时间，使落地更快
                gravity=1.0,  # 增加重力，使玩家更快落地并减少弹跳
                mouse_sensitivity=Vec2(40, 40),  # 调整鼠标灵敏度
                fall_after_jump=True,  # 确保跳跃后立即开始下落
                land_threshold=0.0  # 完全禁用落地弹跳效果
            )
            
            # 禁用玩家控制器的弹跳效果
            if hasattr(player, 'land_anim'):
                player.land_anim = None
            
            # 创建物品栏
            hotbar = Hotbar(Block_list)


            
            # 计算出生点的地形高度
            spawn_x, spawn_z = 0, 0
            spawn_y = floor(noise([spawn_x/scale, spawn_z/scale]) * height_scale + base_height) + 2 # 加2确保在地面之上
            
            # 设置玩家位置
            player.position = (spawn_x, spawn_y, spawn_z)
            player.is_sneaking = False
            player.speed = 5  # 正常速度
            # 不再需要在这里设置jump_height，因为已在构造函数中设置
            
            # 设置游戏状态为正在游戏
            global current_state
            current_state = GameState.PLAYING
            
            # 移除准备就绪提示
            pass
        
        # 在新线程中执行游戏初始化
        Thread(target=start_game_directly).start()
    
    def multiplayer(self):
        # 多人游戏功能（未实现）
        print("多人游戏功能未实现")
    
    def realms(self):
        # Realms功能（未实现）
        print("Minecraft Realms功能未实现")
    
    def options(self):
        # 选项功能（未实现）
        print("选项功能未实现")
    
    def quit_game(self):
        application.quit()

# 创建主菜单
def create_main_menu():
    return MainMenu()

# 启动主菜单
main_menu = create_main_menu()

# 强制应用最高性能优化级别
performance_optimizer.optimization_level = 4
performance_optimizer._apply_optimization_level()

# 添加防止玩家陷入方块的辅助函数
def check_player_stuck_in_blocks():
    """检查玩家是否陷入方块，如果是则将其向上移动"""
    if not player or current_state != GameState.PLAYING:
        return
    
    # 添加静态变量作为冷却计时器，防止频繁触发
    if not hasattr(check_player_stuck_in_blocks, 'cooldown'):
        check_player_stuck_in_blocks.cooldown = 10
    
    # 如果冷却时间未到，则不执行
    if check_player_stuck_in_blocks.cooldown > 0:
        check_player_stuck_in_blocks.cooldown -= 1
        return False
    
    # 获取玩家脚部位置
    feet_pos = Vec3(player.position.x, player.position.y - 0.5, player.position.z)
    
    # 检查玩家脚部是否在方块内
    for chunk_pos in loaded_chunks:
        for block in loaded_chunks[chunk_pos].blocks:
            # 简化的碰撞检测 - 只检查玩家是否在方块内部
            block_pos = block.position
            # 方块的碰撞盒
            block_min = Vec3(block_pos.x - 0.25, block_pos.y - 0.25, block_pos.z - 0.25)
            block_max = Vec3(block_pos.x + 0.25, block_pos.y + 0.25, block_pos.z + 0.25)
            
            # 检查玩家脚部是否在方块内
            if (block_min.x <= feet_pos.x <= block_max.x and
                block_min.y <= feet_pos.y <= block_max.y and
                block_min.z <= feet_pos.z <= block_max.z):
                # 玩家陷入方块，将其向上移动，但幅度减小
                player.y += 0.1
                # 设置冷却时间，防止频繁触发
                check_player_stuck_in_blocks.cooldown = 10
                return True
    
    return False

# 添加检测玩家落地并禁用弹跳的函数
def check_player_landing():
    """检测玩家落地并禁用任何弹跳效果"""
    if not player or current_state != GameState.PLAYING:
        return
    
    # 初始化上一帧的垂直速度变量
    if not hasattr(check_player_landing, 'last_y_velocity'):
        check_player_landing.last_y_velocity = 0
    
    # 计算当前垂直速度
    if not hasattr(player, 'y_last_frame'):
        player.y_last_frame = player.y
    
    current_y_velocity = (player.y - player.y_last_frame) / time.dt
    player.y_last_frame = player.y
    
    # 检测是否落地（从下落状态变为停止或上升状态）
    if check_player_landing.last_y_velocity < -0.1 and current_y_velocity >= -0.05:
        # 玩家已落地，禁用任何弹跳动画
        if hasattr(player, 'y_animator') and player.y_animator:
            player.y_animator.kill()
        
        # 确保玩家不会弹起
        if hasattr(player, 'land_anim'):
            player.land_anim = None
    
    # 更新上一帧的垂直速度
    check_player_landing.last_y_velocity = current_y_velocity

def update():
    global hand, frame_count, last_time, fps, player, sync_chunk_generation_counter, ui_hidden
    try:
        # 每帧重置同步区块生成计数器
        sync_chunk_generation_counter = 0
        
        # 帧率控制 - 计算当前帧率和帧间时间
        current_time = time.time()
        dt = current_time - last_time
        frame_count += 1
        
        # 性能监控 - 测量帧处理时间
        frame_start_time = time.time()
        
        # 更新性能优化系统 - 使用静态变量控制更新频率
        if not hasattr(update, 'optimizer_update_counter'):
            update.optimizer_update_counter = 0
        
        # 处理异步任务完成 - 算法级优化
        completed_tasks = async_manager.check_completed_tasks()
        for task_id, result in completed_tasks:
            if result and task_id.startswith("chunk_"):
                # 处理完成的区块加载
                pass
        
        # 批量渲染优化 - 减少渲染调用
        if player:
            rendered_count = batch_renderer.render_batches(player.position)
            # 更新渐进式加载系统
            progressive_loading_update(player, dt)
        
        # 使用配置化的性能优化系统更新间隔
        update.optimizer_update_counter += 1
        if update.optimizer_update_counter % perf_config.PERFORMANCE_OPTIMIZER_UPDATE_INTERVAL == 0:
            performance_optimizer.update()
        
        # 使用高性能渲染器替代复杂的面片渲染系统
        if camera and hasattr(camera, 'position'):
            high_perf_renderer.fast_render(camera.position)
        
        # 使用配置化的备份渲染间隔，极大减少CPU负担
        if not hasattr(update, 'backup_render_counter'):
            update.backup_render_counter = 0
        
        update.backup_render_counter += 1
        if update.backup_render_counter % perf_config.BACKUP_RENDER_INTERVAL == 0 and 'mesh_renderer' in globals() and mesh_renderer:
            # 备份渲染：更新视锥体剔除
            mesh_renderer.update_culling(camera.position, camera.rotation)
            # 备份渲染：渲染面片
            mesh_renderer.render_faces()
        
        # 每2秒更新一次FPS计数 - 极限降低更新频率以减少开销
        if current_time - last_time >= 2.0:
            fps = frame_count / 2  # 除以2因为是2秒的计数
            frame_count = 0
            last_time = current_time
            
            # 记录帧率到性能优化器
            performance_optimizer.stats['current_fps'] = fps
            
            # 更新FPS显示 - 使用静态变量控制更新频率
            if not hasattr(update, 'fps_update_counter'):
                update.fps_update_counter = 0
            
            # 每16秒更新一次FPS显示，极大减少UI更新开销
            update.fps_update_counter += 1
            if update.fps_update_counter >= 8:
                update.fps_update_counter = 0
                # 更新FPS显示
                if fps_text:
                    fps_text.text = f'FPS: {fps:.0f}'
                
                # 更新性能统计显示
                update_performance_stats()

                # 根据ui_hidden状态控制性能统计显示
                if performance_stats_text:
                    performance_stats_text.enabled = show_performance_stats and not ui_hidden
            
            # 每32秒更新一次优化管理器，极大减少CPU负担
            if not hasattr(update, 'optimization_manager_counter'):
                update.optimization_manager_counter = 0
            
            update.optimization_manager_counter += 1
            if update.optimization_manager_counter >= 16:
                update.optimization_manager_counter = 0
                # 更新优化管理器
                # 检查loaded_chunks是否为空，而不是使用不存在的self.blocks
                if player and loaded_chunks:
                    # 简化的优化管理器更新，只传递玩家位置
                    optimization_manager.update(player, [])
                elif player:
                    optimization_manager.update(player, [])
                
        # 记录帧处理时间
        frame_time_ms = (time.time() - frame_start_time) * 1000
        performance_optimizer.stats['frame_time_ms'] = frame_time_ms

        
        # 根据游戏状态执行不同的更新逻辑
        if current_state == GameState.PLAYING and player:
            try:
                # 注释掉这段代码，因为它可能导致玩家持续弹跳
                # 原来的代码在玩家下落时添加向上的力，这会导致玩家无法稳定落地
                # if hasattr(player, 'y_animator') and player.y_animator.playing and player.y_animator.value < 0:
                #     # 当玩家在下落过程中，增加额外的向上偏移，防止陷入方块
                #     player.y += 0.05 * time.dt
                
                # 禁用任何可能的弹跳动画
                if hasattr(player, 'y_animator') and player.y_animator:
                    # 如果玩家正在向上移动（可能是弹跳），立即停止动画
                    # 检查动画是否仍在播放，而不是访问不存在的.value属性
                    if hasattr(player.y_animator, 'playing') and player.y_animator.playing:
                        player.y_animator.kill()
                
                # 使用静态变量控制物理检测频率
                if not hasattr(update, 'physics_check_counter'):
                    update.physics_check_counter = 0
                
                # 增加计数器
                update.physics_check_counter += 1
                
                # 使用配置化的物理检测间隔，大幅减少CPU负担
                if update.physics_check_counter % perf_config.PLAYER_STUCK_CHECK_INTERVAL == 0:
                    # 检查玩家是否陷入方块，如果是则将其向上移动
                    check_player_stuck_in_blocks()
                
                # 使用配置化的落地检测间隔，大幅减少CPU负担
                if update.physics_check_counter % perf_config.PLAYER_LANDING_CHECK_INTERVAL == 0:
                    # 检测玩家落地并禁用弹跳效果
                    check_player_landing()
                
                # 检查玩家当前区块是否已加载，如果未加载则阻止移动
                if player and hasattr(player, 'position'):
                    current_chunk = get_chunk_position(player.position)
                    next_pos = Vec3(
                        player.position.x + (player.velocity.x if hasattr(player, 'velocity') else 0) * time.dt,
                        player.position.y + (player.velocity.y if hasattr(player, 'velocity') else 0) * time.dt,
                        player.position.z + (player.velocity.z if hasattr(player, 'velocity') else 0) * time.dt
                    )
                    next_chunk = get_chunk_position(next_pos)
                    
                    # 如果下一个位置的区块未加载，阻止移动
                    if next_chunk not in loaded_chunks:
                        player.velocity = Vec3(0, 0, 0)
                        player.gravity = 0
                    else:
                        # 在已加载区块中恢复正常移动
                        player.gravity = 1
                        if not hasattr(player, 'current_gravity'):
                            player.current_gravity = 0
                        
                        # 逐渐恢复到正常重力
                        target_gravity = 1
                        if player.current_gravity < target_gravity:
                            player.current_gravity = min(player.current_gravity + 0.1, target_gravity)
                        player.gravity = player.current_gravity
                
                # Ursina 的 FirstPersonController 会自动处理碰撞和重力

                # 限制手部动画更新 - 只在需要时更新
                if hand and (held_keys['left mouse'] or held_keys['right mouse']):
                    hand.active()
                elif hand:
                    hand.pass_tive()
                
                # 使用静态变量控制区块管理频率
                if not hasattr(update, 'chunk_update_counter'):
                    update.chunk_update_counter = 0
                
                # 增加计数器，控制区块管理频率
                update.chunk_update_counter += 1
                
                # 只在特定帧执行区块管理，大幅减少CPU负担
                # 将区块管理分散到不同帧，避免单帧负载过高
                
                # 使用配置化的区块更新间隔，大幅减少CPU负担
                if update.chunk_update_counter % perf_config.CHUNK_UPDATE_INTERVAL == 0 and player and hasattr(player, 'position'):
                    # 缓存玩家区块位置，避免重复计算
                    player_chunk = get_chunk_position(player.position)
                    
                    # 使用静态变量记录上次位置，只在玩家移动到新区块时才触发更新
                    if not hasattr(update, 'last_player_chunk'):
                        update.last_player_chunk = player_chunk
                    
                    # 只在玩家移动到新区块时才执行复杂的区块管理
                    if update.last_player_chunk != player_chunk:
                        update.last_player_chunk = player_chunk
                        
                        # 使用空间哈希快速查找附近区块
                        nearby_chunks = spatial_hash.query_nearby(player.position, radius=1)
                        
                        # 异步加载必要的区块
                        for dx in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                chunk_pos = (player_chunk[0] + dx, player_chunk[1] + dz)
                                if chunk_pos not in loaded_chunks:
                                    task_id = f"chunk_{chunk_pos[0]}_{chunk_pos[1]}"
                                    async_manager.submit_task(task_id, lambda pos=chunk_pos: pos)
                        
                        # 跳过原有的复杂区块加载逻辑
                        pass
                    
                    # 预加载玩家下方区块 - 这是防止掉落的关键
                    # 每次检查都预加载下方区块，不管玩家是否移动
                    # 减少检查深度，只关注最近的几个区块
                    for depth in range(1, 10, 5):  # 检查玩家下方更少位置，间隔更大
                        below_pos = get_chunk_position(Vec3(player.position.x, player.position.y - depth, player.position.z))
                        if below_pos not in loaded_chunks and below_pos not in preload_queue:
                            # 将下方区块添加到预加载队列，优先级最高
                            preload_queue.insert(0, below_pos)  # 插入到队列开头
                            
                            # 对于最近的几个下方区块，立即同步生成
                            if depth <= 5:  # 只对最近的1个下方区块进行同步生成
                                try:
                                    # 同步生成玩家正下方区块
                                    chunk = Chunk(below_pos)
                                    chunk.generate()
                                    loaded_chunks[below_pos] = chunk
                                    # 强制立即生成
                                    chunk.is_generated = True
                                    # 从预加载队列中移除已生成的区块
                                    if below_pos in preload_queue:
                                        preload_queue.remove(below_pos)
                                except Exception as e:
                                    print(f"Error generating chunk below player: {e}")
                    
                    # 玩家移动到新区块时才更新水平方向的区块加载
                    if update.last_player_chunk != player_chunk:
                        update.last_player_chunk = player_chunk
                        
                        # 增加预加载距离，确保玩家有足够的活动空间
                        preload_distance = 2  # 从1增加到2
                        
                        # 计算玩家朝向，优先加载玩家面向的区块
                        facing_x = int(round(sin(player.rotation_y * pi/180)))
                        facing_z = int(round(cos(player.rotation_y * pi/180)))
                        
                        # 使用集合跟踪已在队列中的区块，提高查找效率
                        if not hasattr(update_chunks, 'queued_chunks'):
                            update_chunks.queued_chunks = set()
                            
                        # 将需要加载的区块添加到优先级队列
                        player_chunk_x, player_chunk_z = player_chunk
                        
                        # 定义不同区域的优先级
                        PRIORITY_PLAYER = 0
                        PRIORITY_FRONT = 1
                        PRIORITY_FRONT_FAR = 1.5
                        PRIORITY_NEAR = 2
                        
                        # 添加区块到队列的辅助函数
                        def add_to_queue(pos, base_priority):
                            if pos not in loaded_chunks and pos not in update_chunks.queued_chunks:
                                # 计算最终优先级（考虑距离和朝向）
                                dx = pos[0] - player_chunk_x
                                dz = pos[1] - player_chunk_z
                                dist = abs(dx) + abs(dz)
                                direction_weight = 1
                                if dx * facing_x + dz * facing_z > 0: # 在前方
                                    direction_weight = 2.5
                                # 确保 direction_weight 不为零
                                if direction_weight == 0:
                                    direction_weight = 0.001  # 使用一个小的非零值
                                priority = base_priority + dist / direction_weight
                                chunk_load_queue.put((priority, pos))
                                update_chunks.queued_chunks.add(pos)

                        # 添加玩家所在区块
                        add_to_queue(player_chunk, PRIORITY_PLAYER)
                        
                        # 添加玩家前方区块
                        front_chunk = (player_chunk_x + facing_x, player_chunk_z + facing_z)
                        add_to_queue(front_chunk, PRIORITY_FRONT)
                        
                        # 添加玩家前方更远的区块
                        front_far_chunk = (player_chunk_x + facing_x*2, player_chunk_z + facing_z*2)
                        add_to_queue(front_far_chunk, PRIORITY_FRONT_FAR)
                        
                        # 添加其他相邻区块
                        for dx in range(-preload_distance, preload_distance+1):
                            for dz in range(-preload_distance, preload_distance+1):
                                chunk_pos = (player_chunk_x + dx, player_chunk_z + dz)
                                # 跳过已明确添加的区块
                                if chunk_pos == player_chunk or chunk_pos == front_chunk or chunk_pos == front_far_chunk:
                                    continue
                                add_to_queue(chunk_pos, PRIORITY_NEAR)
                
                                
                # 异步区块加载 - 使用优先级队列
                # 每帧尝试加载指定数量的区块
                chunks_loaded_this_frame = 0
                while chunks_loaded_this_frame < MAX_CHUNKS_PER_UPDATE and not chunk_load_queue.empty():
                    try:
                        priority, chunk_pos = chunk_load_queue.get_nowait() # 从优先级队列获取
                        
                        # 从跟踪集合中移除 (如果存在)
                        if chunk_pos in update_chunks.queued_chunks:
                            update_chunks.queued_chunks.remove(chunk_pos)
                        
                        # 如果区块尚未加载
                        if chunk_pos not in loaded_chunks:
                            # 检查是否是玩家下方区块，扩大检测范围
                            is_below_player = False
                            if player and hasattr(player, 'position'):
                                # 检查多个下方位置
                                for depth in range(1, 20, 4):
                                    player_below_pos = get_chunk_position(Vec3(player.position.x, player.position.y - depth, player.position.z))
                                    if chunk_pos == player_below_pos:
                                        is_below_player = True
                                        break
                            
                            # 对于玩家下方区块或高优先级区块，直接在主线程中同步生成
                            # 优先级值越小越高，小于等于 PRIORITY_FRONT_FAR (1.5) 的视为高优先级
                            if is_below_player or priority <= 1.5:
                                try:
                                    # 同步生成区块
                                    chunk = Chunk(chunk_pos)
                                    # chunk.generate() # generate 会自动判断同步/异步
                                    loaded_chunks[chunk_pos] = chunk
                                    # 确保生成完成
                                    if not chunk.is_generated:
                                         # 如果 generate 内部判断为异步，这里需要确保它完成
                                         # 这部分逻辑可能需要调整 Chunk.generate() 的实现
                                         # 或者接受这里的潜在延迟
                                         pass 
                                    chunks_loaded_this_frame += 1
                                except Exception as e:
                                    logging.error(f"Error generating high-priority chunk {chunk_pos}: {e}", exc_info=True)
                            else:
                                # 其他区块使用线程池异步加载
                                # 注意：executor.submit 不直接接受优先级，Chunk.generate() 内部处理异步
                                if chunk_pos not in loaded_chunks: # 再次检查，避免重复提交
                                     chunk = Chunk(chunk_pos) # 创建 Chunk 实例，其 generate 会被调用
                                     loaded_chunks[chunk_pos] = chunk # 立即添加到 loaded_chunks，防止重复加载
                                     # generate 方法内部会处理异步提交
                                     chunks_loaded_this_frame += 1
                                     
                    except Empty: # 捕获队列为空的异常
                        break
                    except Exception as e:
                        logging.error(f"Error processing chunk queue: {e}", exc_info=True)
                        # 确保异常时也移除跟踪 (如果 chunk_pos 已定义)
                        try:
                            if 'chunk_pos' in locals() and chunk_pos in update_chunks.queued_chunks:
                                update_chunks.queued_chunks.remove(chunk_pos)
                        except Exception as remove_err:
                             logging.error(f"Error removing chunk {chunk_pos} from queued_chunks after queue error: {remove_err}")
                
                # 区块卸载 - 每30帧执行一次
                if update.chunk_update_counter % 30 == 15 and player and hasattr(player, 'position'):
                    player_chunk = get_chunk_position(player.position)
                    player_chunk_x, player_chunk_z = player_chunk
                    
                    # 使用静态变量控制卸载检查的区块索引
                    if not hasattr(update, 'unload_index'):
                        update.unload_index = 0
                    
                    # 获取已加载区块列表
                    chunk_keys = list(loaded_chunks.keys())
                    if chunk_keys:  # 确保有区块可检查
                        # 每次只检查一个区块，分散卸载压力
                        if update.unload_index >= len(chunk_keys):
                            update.unload_index = 0
                        
                        # 获取当前要检查的区块
                        chunk_pos = chunk_keys[update.unload_index]
                        update.unload_index = (update.unload_index + 1) % len(chunk_keys)
                        
                        # 计算与玩家的距离
                        dist = abs(chunk_pos[0] - player_chunk_x) + abs(chunk_pos[1] - player_chunk_z)
                        # 使用更大的缓冲区，减少频繁加载/卸载同一区块
                        if dist > RENDER_DISTANCE + 3:  # 增加缓冲区大小
                            if chunk_pos in loaded_chunks:  # 再次检查，以防在其他线程中被修改
                                try:
                                    chunk_to_unload = loaded_chunks[chunk_pos]
                                    # 从空间网格移除方块
                                    # 确保 chunk_to_unload.blocks 存在且可迭代
                                    if hasattr(chunk_to_unload, 'blocks') and chunk_to_unload.blocks:
                                        # 创建副本进行迭代，防止在迭代过程中修改列表
                                        blocks_to_remove = list(chunk_to_unload.blocks)
                                        for block in blocks_to_remove:
                                            if block and hasattr(block, 'position'): # 确保 block 有效且有 position
                                                spatial_grid.remove_block(block)
                                            else:
                                                logging.warning(f"Invalid block found during chunk unload: {block} in chunk {chunk_pos}")
                                    # 销毁区块实体
                                    chunk_to_unload.destroy()
                                    del loaded_chunks[chunk_pos]
                                    # 从 queued_chunks 中也移除 (如果存在)
                                    if hasattr(update_chunks, 'queued_chunks') and chunk_pos in update_chunks.queued_chunks:
                                        try:
                                            update_chunks.queued_chunks.remove(chunk_pos)
                                        except KeyError:
                                            pass # 可能已经被其他地方移除
                                except KeyError: # 处理 chunk_pos 可能已被删除的情况
                                    logging.warning(f"Chunk {chunk_pos} already removed before unloading.")
                                except Exception as e:
                                    logging.error(f"Error unloading chunk at {chunk_pos}: {e}", exc_info=True)
                
                # 重置计数器，避免溢出
                if update.chunk_update_counter >= 120:  # 每120帧重置一次
                    update.chunk_update_counter = 0
                    
            except Exception as e:
                # 添加更详细的日志记录
                logging.error(f"Error in game update loop (inner try - state: {current_state}, player: {'exists' if player else 'None'}): {e}", exc_info=True)
        
        # 限制区块更新频率 - 使用时间间隔控制
        if current_state == GameState.PLAYING:
            if not hasattr(update, 'last_chunk_update'):
                update.last_chunk_update = 0
            
            # 增加更新间隔，减少CPU负担
            if current_time - update.last_chunk_update >= UPDATE_INTERVAL * 3:  # 进一步增加间隔
                try:
                    # 使用批处理方式更新区块，减少每帧工作量
                    update_chunks()
                except Exception as e:
                    logging.error(f"Error calling update_chunks: {e}", exc_info=True)
                update.last_chunk_update = current_time
        
        # 帧率限制 - 确保不超过目标帧率
        elapsed = time.time() - current_time
        if elapsed < frame_duration:
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        logging.critical(f"Critical error in main update function: {e}", exc_info=True)
        # 出现严重错误时，尝试恢复游戏状态
        if not hasattr(update, 'error_count'):
            update.error_count = 0
        
        update.error_count += 1
        # 如果连续出现多次错误，尝试重置游戏状态
        if update.error_count > 10:
            update.error_count = 0
            print("Attempting to recover game state...")
            # 重置关键状态变量
            if hasattr(update, 'last_chunk_update'):
                update.last_chunk_update = 0
            if hasattr(update, 'last_player_chunk'):
                update.last_player_chunk = None

# 添加LOD（细节层次）系统函数
def apply_lod_settings():
    """根据距离应用不同的LOD设置"""
    if not hasattr(player, 'position'):
        return
    
    # 限制LOD更新频率
    current_time = time.time()
    if not hasattr(apply_lod_settings, 'last_update'):
        apply_lod_settings.last_update = 0
    
    # 每0.5秒更新一次LOD
    if current_time - apply_lod_settings.last_update < 0.5:
        return
    
    apply_lod_settings.last_update = current_time
    
    # 定义LOD距离阈值 (根据 question.md)
    LOD0_DIST = 15  # 近距离：完整细节
    LOD1_DIST = 30  # 中距离：简化处理
    # LOD2_DIST: > 30m 远距离：最低细节
    
    # 优化：缓存区块距离，避免重复计算
    if not hasattr(apply_lod_settings, 'chunk_distances'):
        apply_lod_settings.chunk_distances = {}
        
    # 优化：分批处理区块，避免单帧负担过重
    if not hasattr(apply_lod_settings, 'chunk_index'):
        apply_lod_settings.chunk_index = 0
        
    chunk_keys = list(loaded_chunks.keys())
    if not chunk_keys:
        return
        
    # 每帧处理一部分区块 (例如 1/5)
    batch_size = max(1, len(chunk_keys) // 5)
    start_index = apply_lod_settings.chunk_index
    end_index = (start_index + batch_size) % len(chunk_keys)
    
    chunks_to_process = []
    if start_index < end_index:
        chunks_to_process = chunk_keys[start_index:end_index]
    else: # 处理跨越列表末尾的情况
        chunks_to_process = chunk_keys[start_index:] + chunk_keys[:end_index]
        
    apply_lod_settings.chunk_index = end_index

    for chunk_pos in chunks_to_process:
        if chunk_pos not in loaded_chunks: # 检查区块是否仍然存在
            continue
        chunk = loaded_chunks[chunk_pos]
        
        # 计算或获取缓存的距离
        # 优化：仅在玩家移动较多时重新计算所有距离
        # 这里简化为每次都计算当前批次的距离
        chunk_center_x = chunk_pos[0] * CHUNK_SIZE + CHUNK_SIZE / 2
        chunk_center_z = chunk_pos[1] * CHUNK_SIZE + CHUNK_SIZE / 2
        # 优化：只考虑水平距离，忽略Y轴差异
        dist = distance((player.position.x, player.position.z), (chunk_center_x, chunk_center_z))
        apply_lod_settings.chunk_distances[chunk_pos] = dist
        
        # 根据距离设置LOD级别
        if dist < LOD0_DIST:  # 近距离
            lod_level = 0
        elif dist < LOD1_DIST:  # 中距离
            lod_level = 1
        else:  # 远距离
            lod_level = 2
        
        # 应用LOD设置到区块中的方块
        for block in chunk.blocks:
            # 优化：跳过非Block对象或已隐藏的对象
            if not isinstance(block, Block) or not block.enabled:
                continue
                
            # 根据LOD级别调整方块属性
            if lod_level == 0:
                # LOD 0: 近距离 (<15m) - 完整细节
                block.collision = True
                block.visible = True
                # block.model = 'assets/block.obj' # 确保使用完整模型 (如果未来有简化模型)
            elif lod_level == 1:
                # LOD 1: 中距离 (15-30m) - 简化处理
                block.collision = True # 保持碰撞
                block.visible = True
                # block.model = 'assets/block_simple.obj' # 未来可切换到简化模型
            else:  # lod_level == 2
                # LOD 2: 远距离 (>30m) - 最低细节
                block.collision = False # 禁用碰撞
                block.visible = True # 保持可见 (未来可切换到公告板)
                # block.model = 'assets/block_billboard.obj' # 未来可切换到公告板模型

# 在update函数中调用LOD系统、动态渲染距离调整和综合性能优化器
def update_with_lod():
    global RENDER_DISTANCE, last_render_distance_update
    
    # 获取当前时间和帧率
    current_time = time.time()
    from ursina import application
    current_fps = getattr(application, 'fps', 30)
    
    # 更新综合性能优化器和GPU优化系统
    if current_state == GameState.PLAYING and hasattr(player, 'position'):
        # 更新综合性能优化器
        comprehensive_optimizer.update(player.position, application.dt)
        
        # 更新GPU优化系统
        gpu_optimization_integrator.update(player.position, application.dt)
        
        # 更新GPU视锥体剔除
        if gpu_optimization_integrator.use_gpu_frustum_culling:
            gpu_frustum_culling.update(player.position, player.camera_pivot.rotation)
        
        # 更新输入优化器
        input_optimizer.update(player, application.dt)
    
    # 调用原始update函数
    update()
    
    # 应用LOD设置
    if current_state == GameState.PLAYING and hasattr(player, 'position'):
        apply_lod_settings()
    
    # 动态调整渲染距离 - 使用更激进的策略
    if DYNAMIC_RENDER_DISTANCE and current_state == GameState.PLAYING:
        # 每隔一段时间更新一次渲染距离
        if current_time - last_render_distance_update > RENDER_DISTANCE_UPDATE_INTERVAL:
            last_render_distance_update = current_time
            
            # 根据帧率动态调整渲染距离 - 更激进的策略
            if current_fps > 55:  # 帧率非常高
                # 帧率非常高，可以适度增加渲染距离
                RENDER_DISTANCE = min(MAX_RENDER_DISTANCE, RENDER_DISTANCE + 1)
            elif current_fps > 45:  # 帧率高
                # 帧率高，保持当前渲染距离
                pass
            elif current_fps < 30:  # 帧率低
                # 帧率低，减少渲染距离
                RENDER_DISTANCE = max(1, RENDER_DISTANCE - 1)
            elif current_fps < 20:  # 帧率非常低
                # 帧率非常低，立即降低到最小渲染距离
                RENDER_DISTANCE = 1
                # 同时提高综合优化器的优化级别
                comprehensive_optimizer.set_optimization_level(4)  # 高性能模式
            
            # 记录日志
            logging.info(f"动态调整渲染距离: {RENDER_DISTANCE} (FPS: {current_fps:.1f})")

# 替换原始update函数
app.update = update_with_lod

# 初始化线程池和锁
chunk_executor = ThreadPoolExecutor(max_workers=2)
chunk_lock = Lock()
block_type_id = 0

# 初始化综合性能优化器
def initialize_gpu_optimization():
    # 配置GPU优化集成器
    gpu_optimization_integrator._init_systems()
    gpu_optimization_integrator.use_gpu_frustum_culling = True
    gpu_optimization_integrator.use_gpu_instancing = True
    gpu_optimization_integrator.use_shader_optimization = True
    gpu_optimization_integrator.adaptive_optimization = True
    gpu_optimization_integrator.target_fps = 60
    
    # 注册GPU优化子系统
    comprehensive_optimizer.register_subsystem('gpu_frustum_culling', gpu_frustum_culling)
    comprehensive_optimizer.register_subsystem('gpu_optimization', gpu_optimization_manager)
    
    logging.info("GPU优化系统初始化完成")

def initialize_comprehensive_optimizer():
    # 配置综合性能优化器
    comprehensive_optimizer._init_systems()  # 使用已有的初始化方法
    comprehensive_optimizer.target_fps = 60  # 设置目标帧率
    comprehensive_optimizer.min_acceptable_fps = 20  # 设置最低可接受帧率（从30降低到20）
    comprehensive_optimizer.adaptive_mode = False  # 关闭自适应模式
    comprehensive_optimizer.optimization_level = 5  # 设置为最高优化级别（极限性能模式）
    comprehensive_optimizer.thread_pool_size = max(2, os.cpu_count() - 2) if os.cpu_count() else 2  # 减少线程数，避免CPU过载
    comprehensive_optimizer.use_gpu_instancing = True  # 启用GPU实例化渲染
    comprehensive_optimizer.use_mesh_batching = True  # 启用网格批处理
    comprehensive_optimizer.use_occlusion_culling = True  # 启用遮挡剔除
    comprehensive_optimizer.use_shader_optimization = True  # 启用着色器优化
    comprehensive_optimizer.use_texture_compression = True  # 启用纹理压缩
    comprehensive_optimizer.use_adaptive_resolution = True  # 启用自适应分辨率
    comprehensive_optimizer.extreme_mode = True  # 启用极限模式
    comprehensive_optimizer.extreme_lod_distance = 2  # 极限LOD距离（从3降低到2）
    comprehensive_optimizer.extreme_render_distance = 1  # 极限渲染距离
    
    # 初始化GPU优化系统
    initialize_gpu_optimization()
    
    # 配置GPU相关选项
    comprehensive_optimizer.use_gpu_acceleration = True
    comprehensive_optimizer.use_compute_shaders = True
    comprehensive_optimizer.use_geometry_instancing = True
    
    # 注册优化子系统
    comprehensive_optimizer.register_subsystem('frustum_culling', frustum_culling_manager)
    comprehensive_optimizer.register_subsystem('lod_system', lod_manager)
    comprehensive_optimizer.register_subsystem('performance_optimizer', performance_optimizer)
    comprehensive_optimizer.register_subsystem('chunk_loading_optimizer', chunk_loading_optimizer)
    
    logging.info("综合性能优化器初始化完成")

# 在游戏启动时初始化综合性能优化器
initialize_comprehensive_optimizer()

# 创建性能统计显示
def create_performance_stats_display():
    global performance_stats_text
    if performance_stats_text:
        destroy(performance_stats_text)
    
    performance_stats_text = Text(
        text='',
        position=(0.6, 0.45),
        scale=1,
        color=color.white,
        background=True,
        background_color=color.black66,
        visible=show_performance_stats
    )

# 更新性能统计显示
def update_performance_stats():
    global performance_stats_text, fps
    if not performance_stats_text or not show_performance_stats:
        return
    
    # 获取性能优化器的统计信息
    stats = performance_optimizer.stats
    
    # 构建统计信息文本
    stats_text = f"FPS: {fps:.1f}\n"
    
    # 添加GPU优化状态
    stats_text += f"\nGPU优化状态:\n"
    stats_text += f"GPU视锥体剔除: {'开启' if gpu_optimization_integrator.use_gpu_frustum_culling else '关闭'}\n"
    stats_text += f"GPU实例化渲染: {'开启' if gpu_optimization_integrator.use_gpu_instancing else '关闭'}\n"
    stats_text += f"着色器优化: {'开启' if gpu_optimization_integrator.use_shader_optimization else '关闭'}\n"
    stats_text += f"自适应优化: {'开启' if gpu_optimization_integrator.adaptive_optimization else '关闭'}\n"
    
    # 添加GPU性能指标
    gpu_stats = gpu_optimization_integrator.get_gpu_stats()
    stats_text += f"GPU渲染时间: {gpu_stats.get('render_time_ms', 0):.1f}ms\n"
    stats_text += f"实例化实体数: {gpu_stats.get('instanced_entities', 0)}\n"
    stats_text += f"优化实体数: {gpu_stats.get('optimized_entities', 0)}\n"
    stats_text += f"帧处理时间: {stats['frame_time_ms']:.1f}ms\n"
    stats_text += f"总方块数: {stats['total_blocks']}\n"
    stats_text += f"可见方块: {stats['visible_blocks']}\n"
    stats_text += f"剔除方块: {stats['culled_blocks']}\n"
    stats_text += f"高细节: {stats['high_detail_blocks']}\n"
    stats_text += f"中细节: {stats['medium_detail_blocks']}\n"
    stats_text += f"低细节: {stats['low_detail_blocks']}\n"
    
    # 添加综合性能优化器状态
    stats_text += f"\n综合性能优化器:\n"
    stats_text += f"优化级别: {comprehensive_optimizer.get_optimization_level_name()}\n"
    stats_text += f"自适应模式: {'开启' if comprehensive_optimizer.adaptive_mode else '关闭'}\n"
    stats_text += f"目标帧率: {comprehensive_optimizer.target_fps} FPS\n"
    stats_text += f"多线程: {'开启' if comprehensive_optimizer.enable_multithreading else '关闭'}\n"
    stats_text += f"内存优化: {'开启' if comprehensive_optimizer.enable_memory_optimization else '关闭'}\n"
    stats_text += f"极限模式: {'开启' if comprehensive_optimizer.extreme_mode_enabled else '关闭'}\n"
    
    # 添加优化管理器状态
    stats_text += f"\n优化管理器状态:\n"
    stats_text += f"实例化渲染(F1): {'开启' if optimization_manager.use_instanced_rendering else '关闭'}\n"
    stats_text += f"网格合并(F2): {'开启' if optimization_manager.use_mesh_combining else '关闭'}\n"
    stats_text += f"距离剔除(F3): {'开启' if optimization_manager.use_distance_culling else '关闭'}\n"
    stats_text += f"区块管理(F4): {'开启' if optimization_manager.use_chunk_management else '关闭'}\n"
    stats_text += f"自适应优化(F5): {'开启' if optimization_manager.adaptive_optimization else '关闭'}"
    
    # 添加区块加载优化器统计
    if hasattr(chunk_loading_optimizer, 'get_loading_stats'):
        loading_stats = chunk_loading_optimizer.get_loading_stats()
        stats_text += f"\n区块加载优化:\n"
        stats_text += f"已加载区块: {loading_stats.get('loaded_chunks', 0)}\n"
        stats_text += f"加载中区块: {loading_stats.get('loading_chunks', 0)}\n"
        stats_text += f"缓存命中率: {loading_stats.get('cache_hit_rate', 0):.1%}\n"
        stats_text += f"平均加载时间: {loading_stats.get('avg_load_time', 0)*1000:.1f}ms\n"
    
    # 添加渐进式加载系统统计
    if hasattr(progressive_loader, 'get_loading_stats'):
        prog_stats = progressive_loader.get_loading_stats()
        stats_text += f"\n渐进式加载系统:\n"
        stats_text += f"启动阶段: {'进行中' if not prog_stats.get('startup_complete', True) else '完成'} ({prog_stats.get('startup_progress', 1.0)*100:.0f}%)\n"
        stats_text += f"当前阶段: {prog_stats.get('current_stage', '完成')}\n"
        stats_text += f"已加载区块: {prog_stats.get('startup_chunks_loaded', 0)}\n"
        stats_text += f"加载中区块: {prog_stats.get('loading_chunks_count', 0)}\n"
        stats_text += f"加载速率: {prog_stats.get('current_loading_rate', 0)} 区块/帧\n"
        stats_text += f"立即队列: {prog_stats.get('immediate_queue_size', 0)}\n"
        stats_text += f"后台队列: {prog_stats.get('background_queue_size', 0)}\n"
        total_loaded = prog_stats.get('startup_chunks_loaded', 0) + prog_stats.get('immediate_chunks_loaded', 0) + prog_stats.get('background_chunks_loaded', 0)
        stats_text += f"总加载区块: {total_loaded}\n"
    
    # 添加区块缓存统计
    if hasattr(block_cache, 'get_cache_stats'):
        cache_info = block_cache.get_cache_stats()
        stats_text += f"缓存区块: {cache_info.get('total_cache_size', 0)}\n"
        stats_text += f"内存占用: {cache_info.get('memory_usage', 0):.1f}MB\n"
    
    # 显示优化系统状态
    stats_text += f"\n优化系统: {'开启' if performance_optimizer.enabled else '关闭'}\n"
    stats_text += f"自适应优化: {'开启' if performance_optimizer.adaptive_mode else '关闭'}\n"
    
    # 显示优化级别
    optimization_level_names = ['低(高质量)', '中(平衡)', '高(高性能)']
    opt_level = performance_optimizer.optimization_level
    stats_text += f"优化级别: {optimization_level_names[opt_level]}\n"
    
    # 显示动态渲染距离信息
    stats_text += f"渲染距离: {RENDER_DISTANCE} (动态: {'开启' if DYNAMIC_RENDER_DISTANCE else '关闭'})\n"
    stats_text += f"线程池: {chunk_executor._max_workers} 线程\n"
    # 水方块相关代码已移除
    
    # 显示内存使用情况
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        stats_text += f"内存使用: {memory_info.rss / (1024 * 1024):.1f}MB\n"
    except ImportError:
        # 如果没有psutil，使用gc模块获取内存使用情况
        stats_text += f"GC对象数: {len(gc.get_objects())}\n"
    
    # 显示子系统状态
    stats_text += f"视锥体剔除: {'开启' if performance_optimizer.enable_frustum_culling else '关闭'}\n"
    stats_text += f"LOD系统: {'开启' if performance_optimizer.enable_lod else '关闭'}"
    
    # 更新文本
    performance_stats_text.text = stats_text

# 添加性能监控
if __name__ == '__main__':
    print("Starting game with optimized performance...")
    app.run()
