# 视锥体剔除模块
# 用于优化渲染性能，只渲染摄像机视锥体内的方块

from ursina import *
from math import tan, radians
import numpy as np

class Frustum:
    """摄像机视锥体类，用于判断物体是否在视锥体内"""
    
    def __init__(self, camera=None):
        # 延迟初始化camera，避免在游戏启动时camera.main_camera为None的问题
        self._camera = None
        # 视锥体的六个平面 (近、远、左、右、上、下)
        self.planes = [Vec4(0,0,0,0) for _ in range(6)]
        # 标记是否已初始化
        self.initialized = False
    
    @property
    def camera(self):
        # 延迟获取camera对象
        if self._camera is None:
            from ursina import camera
            self._camera = camera
        return self._camera
    
    def update(self):
        """更新视锥体平面参数"""
        # 检查camera是否可用
        if not self.camera or not hasattr(self.camera, 'view_matrix'):
            return False
        
        # 获取摄像机的视图矩阵和投影矩阵
        view_matrix = self.camera.view_matrix
        projection_matrix = self.camera.projection_matrix
        
        # 计算视图投影矩阵
        vp_matrix = projection_matrix * view_matrix
        
        # 提取视锥体平面
        # 左平面
        self.planes[0] = Vec4(
            vp_matrix.row_1.w + vp_matrix.row_1.x,
            vp_matrix.row_2.w + vp_matrix.row_2.x,
            vp_matrix.row_3.w + vp_matrix.row_3.x,
            vp_matrix.row_4.w + vp_matrix.row_4.x
        ).normalized()
        
        # 右平面
        self.planes[1] = Vec4(
            vp_matrix.row_1.w - vp_matrix.row_1.x,
            vp_matrix.row_2.w - vp_matrix.row_2.x,
            vp_matrix.row_3.w - vp_matrix.row_3.x,
            vp_matrix.row_4.w - vp_matrix.row_4.x
        ).normalized()
        
        # 下平面
        self.planes[2] = Vec4(
            vp_matrix.row_1.w + vp_matrix.row_1.y,
            vp_matrix.row_2.w + vp_matrix.row_2.y,
            vp_matrix.row_3.w + vp_matrix.row_3.y,
            vp_matrix.row_4.w + vp_matrix.row_4.y
        ).normalized()
        
        # 上平面
        self.planes[3] = Vec4(
            vp_matrix.row_1.w - vp_matrix.row_1.y,
            vp_matrix.row_2.w - vp_matrix.row_2.y,
            vp_matrix.row_3.w - vp_matrix.row_3.y,
            vp_matrix.row_4.w - vp_matrix.row_4.y
        ).normalized()
        
        # 近平面
        self.planes[4] = Vec4(
            vp_matrix.row_1.w + vp_matrix.row_1.z,
            vp_matrix.row_2.w + vp_matrix.row_2.z,
            vp_matrix.row_3.w + vp_matrix.row_3.z,
            vp_matrix.row_4.w + vp_matrix.row_4.z
        ).normalized()
        
        # 远平面
        self.planes[5] = Vec4(
            vp_matrix.row_1.w - vp_matrix.row_1.z,
            vp_matrix.row_2.w - vp_matrix.row_2.z,
            vp_matrix.row_3.w - vp_matrix.row_3.z,
            vp_matrix.row_4.w - vp_matrix.row_4.z
        ).normalized()
        
        self.initialized = True
        return True
    
    def is_point_visible(self, point):
        """判断点是否在视锥体内"""
        for plane in self.planes:
            # 计算点到平面的距离
            distance = plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w
            if distance < 0:
                return False  # 点在平面外部
        return True  # 点在所有平面内部
    
    def is_sphere_visible(self, center, radius):
        """判断球体是否在视锥体内或与视锥体相交"""
        for plane in self.planes:
            # 计算球心到平面的距离
            distance = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w
            if distance < -radius:
                return False  # 球体完全在平面外部
        return True  # 球体至少部分在视锥体内
    
    def is_box_visible(self, min_point, max_point):
        """判断轴对齐包围盒是否在视锥体内或与视锥体相交"""
        for plane in self.planes:
            # 找到包围盒中到平面最远的点（p-vertex）
            p_vertex = Vec3(
                max_point.x if plane.x >= 0 else min_point.x,
                max_point.y if plane.y >= 0 else min_point.y,
                max_point.z if plane.z >= 0 else min_point.z
            )
            
            # 计算p-vertex到平面的距离
            distance = plane.x * p_vertex.x + plane.y * p_vertex.y + plane.z * p_vertex.z + plane.w
            
            if distance < 0:
                return False  # 包围盒完全在平面外部
        
        return True  # 包围盒至少部分在视锥体内

class FrustumCullingManager:
    """视锥体剔除管理器，用于管理和优化视锥体剔除"""
    
    def __init__(self):
        self.frustum = Frustum()
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒）
        self.culled_entities = set()  # 被剔除的实体
        self.visible_entities = set()  # 可见的实体
        
        # 性能统计
        self.stats = {
            'culled_count': 0,
            'visible_count': 0,
            'total_count': 0,
            'culling_ratio': 0.0,
            'total_blocks': 0,
            'visible_blocks': 0,
            'culled_blocks': 0,
            'frame_time_ms': 0.0
        }
    
    def update(self):
        """更新视锥体和执行剔除"""
        if not self.enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 更新视锥体
        if not self.frustum.update():
            # 如果视锥体更新失败（可能是camera未初始化），则跳过本次更新
            return
        
        # 记录开始时间（用于计算帧时间）
        start_time = time.time()
        
        # 重置统计信息
        self.stats['culled_count'] = 0
        self.stats['visible_count'] = 0
        self.stats['total_count'] = 0
        self.stats['total_blocks'] = 0
        self.stats['visible_blocks'] = 0
        self.stats['culled_blocks'] = 0
        
        # 清空剔除和可见实体集合
        self.culled_entities.clear()
        self.visible_entities.clear()
        
        # 计算帧处理时间（毫秒）
        self.stats['frame_time_ms'] = (time.time() - start_time) * 1000
        
        # 获取所有实体
        entities = [e for e in scene.entities if hasattr(e, 'position') and hasattr(e, 'scale')]
        
        # 限制每次处理的实体数量
        max_entities_per_update = 500  # 每次最多处理500个实体
        entities_to_process = entities[:max_entities_per_update]
        
        # 更新统计信息
        self.stats['total_count'] = len(entities_to_process)
        self.stats['total_blocks'] = len(entities_to_process)
        
        # 执行视锥体剔除
        for entity in entities_to_process:
            # 跳过UI元素和摄像机
            if not hasattr(entity, 'position') or not hasattr(entity, 'scale') or entity == camera:
                continue
            
            # 计算包围盒
            half_size = entity.scale / 2
            min_point = entity.position - half_size
            max_point = entity.position + half_size
            
            # 判断是否在视锥体内
            if self.frustum.is_box_visible(min_point, max_point):
                # 实体在视锥体内，设为可见
                if not entity.visible:
                    entity.visible = True
                self.visible_entities.add(entity)
                self.stats['visible_count'] += 1
                self.stats['visible_blocks'] += 1
            else:
                # 实体在视锥体内，设为不可见
                if entity.visible:
                    entity.visible = False
                self.culled_entities.add(entity)
                self.stats['culled_count'] += 1
                self.stats['culled_blocks'] += 1

    def filter_visible_entities(self, entities):
        """过滤出视锥体内的实体，使用优化算法提高性能"""
        if not self.enabled or self.frustum is None:
            return entities  # 如果剔除功能禁用或frustum未初始化，则返回所有实体
        
        # 测量处理时间
        start_time = time.time()
        
        self.stats['total_blocks'] = len(entities)
        
        # 优化1: 使用空间哈希网格加速剔除过程
        # 将实体按空间位置分组，可以快速跳过整组不在视锥体内的实体
        grid_size = 8  # 网格大小
        spatial_grid = {}
        
        # 将实体放入空间网格
        for entity in entities:
            if not hasattr(entity, 'position') or not hasattr(entity, 'scale'):
                continue
                
            # 计算网格坐标
            grid_x = int(entity.position.x // grid_size)
            grid_y = int(entity.position.y // grid_size)
            grid_z = int(entity.position.z // grid_size)
            grid_key = (grid_x, grid_y, grid_z)
            
            if grid_key not in spatial_grid:
                spatial_grid[grid_key] = []
            spatial_grid[grid_key].append(entity)
        
        # 优化2: 视锥体剔除结果缓存
        # 使用字典缓存最近的剔除结果，避免重复计算
        entity_cache = {}
        current_time = time.time()
        cache_lifetime = 0.1  # 缓存有效期(秒)
        
        visible_entities = []
        
        # 优化3: 批量处理网格单元
        for grid_key, grid_entities in spatial_grid.items():
            # 计算网格单元的边界盒
            min_x = grid_key[0] * grid_size
            min_y = grid_key[1] * grid_size
            min_z = grid_key[2] * grid_size
            max_x = min_x + grid_size
            max_y = min_y + grid_size
            max_z = min_z + grid_size
            
            # 检查网格单元是否在视锥体内
            grid_min = Vec3(min_x, min_y, min_z)
            grid_max = Vec3(max_x, max_y, max_z)
            
            # 如果整个网格单元都在视锥体内，跳过其中所有实体
            if not self.frustum.is_box_visible(grid_min, grid_max):
                self.stats['culled_blocks'] += len(grid_entities)
                continue
            
            # 处理网格内的实体
            for entity in grid_entities:
                # 优化4: 使用实体ID作为缓存键
                entity_id = id(entity)
                
                # 检查缓存
                if entity_id in entity_cache:
                    cache_time, is_visible = entity_cache[entity_id]
                    if current_time - cache_time < cache_lifetime:
                        # 使用缓存结果
                        if is_visible:
                            visible_entities.append(entity)
                        continue
                
                # 优化5: 基于距离的处理策略
                if hasattr(entity, 'distance_to_player'):
                    # 远距离实体使用更激进的剔除策略
                    if entity.distance_to_player > 48:
                        # 距离非常远的实体有75%概率直接剔除
                        if random() > 0.25:
                            entity_cache[entity_id] = (current_time, False)
                            self.stats['culled_blocks'] += 1
                            continue
                    elif entity.distance_to_player > 32:
                        # 距离较远的实体有50%概率直接剔除
                        if random() > 0.5:
                            entity_cache[entity_id] = (current_time, False)
                            self.stats['culled_blocks'] += 1
                            continue
                
                # 计算包围盒
                half_size = entity.scale / 2
                min_point = entity.position - half_size
                max_point = entity.position + half_size
                
                # 判断是否在视锥体内
                is_visible = self.frustum.is_box_visible(min_point, max_point)
                
                # 更新缓存
                entity_cache[entity_id] = (current_time, is_visible)
                
                if is_visible:
                    visible_entities.append(entity)
                    self.stats['visible_blocks'] += 1
                else:
                    self.stats['culled_blocks'] += 1
        
        # 记录处理耗时
        self.stats['frame_time_ms'] = (time.time() - start_time) * 1000
        self.stats['visible_blocks'] = len(visible_entities)
        
        # 计算剔除比例
        if self.stats['total_count'] > 0:
            self.stats['culling_ratio'] = self.stats['culled_count'] / self.stats['total_count']
        
        return visible_entities

    def is_visible(self, entity):
        """判断实体是否在视锥体内"""
        if not self.enabled:
            return True  # 如果未启用剔除，则默认可见
        
        # 如果frustum未初始化，则默认可见
        if not hasattr(self.frustum, 'initialized') or not self.frustum.initialized:
            return True
        
        # 如果实体在可见集合中，返回True
        if entity in self.visible_entities:
            return True
        
        # 如果实体在剔除集合中，返回False
        if entity in self.culled_entities:
            return False
        
        # 如果实体不在任何集合中，进行视锥体测试
        if hasattr(entity, 'position') and hasattr(entity, 'scale'):
            half_size = entity.scale / 2
            min_point = entity.position - half_size
            max_point = entity.position + half_size
            return self.frustum.is_box_visible(min_point, max_point)
        
        # 默认可见（如果实体没有位置/缩放属性）
        return True

# 创建全局视锥体剔除管理器实例
frustum_culling_manager = FrustumCullingManager()

# 扩展空间网格类的方法，集成视锥体剔除功能
def get_visible_blocks(spatial_grid, position, radius=1, use_frustum_culling=True):
    """获取视锥体内的方块"""
    # 先获取附近的所有方块
    nearby_blocks = spatial_grid.get_nearby_blocks(position, radius)
    
    # 如果启用视锥体剔除，过滤出视锥体内的方块
    if use_frustum_culling and frustum_culling_manager.enabled:
        return frustum_culling_manager.filter_visible_entities(nearby_blocks)
    else:
        return nearby_blocks

    def cull_entity(self, entity):
        """对单个实体进行视锥体剔除"""
        if not self.enabled or not hasattr(entity, 'position') or not hasattr(entity, 'scale'):
            return
        
        self.stats['total_count'] += 1
        self.stats['total_blocks'] += 1
        
        # 检查实体是否在视锥体内
        is_visible = self.is_visible(entity)
        
        if is_visible:
            # 实体在视锥体内，设为可见
            if not entity.visible:
                entity.visible = True
            self.visible_entities.add(entity)
            self.stats['visible_count'] += 1
            self.stats['visible_blocks'] += 1
        else:
            # 实体在视锥体内，设为不可见
            if entity.visible:
                entity.visible = False
            self.culled_entities.add(entity)
            self.stats['culled_count'] += 1
            self.stats['culled_blocks'] += 1