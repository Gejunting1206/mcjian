# 视锥体剔除模块
# 用于优化渲染性能，只渲染摄像机视锥体内的方块

from ursina import *
from math import tan, radians
import numpy as np

class Frustum:
    """摄像机视锥体类，用于判断物体是否在视锥体内"""
    
    def __init__(self, camera=None):
        self.camera = camera or camera.main_camera
        # 视锥体的六个平面 (近、远、左、右、上、下)
        self.planes = [Vec4(0,0,0,0) for _ in range(6)]
        # 更新视锥体平面
        self.update()
    
    def update(self):
        """更新视锥体平面参数"""
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

# 视锥体剔除管理器
class FrustumCullingManager:
    def __init__(self, camera=None):
        self.camera = camera
        self.frustum = None  # 延迟初始化Frustum对象
        self.update_interval = 0.1  # 视锥体更新间隔（秒）- 增加间隔减少计算频率
        self.last_update_time = 0
        self.enabled = True
        self.adaptive_culling = True  # 启用自适应剔除
        self.culling_radius = 64  # 最大剔除距离
        self.stats = {
            'total_blocks': 0,
            'visible_blocks': 0,
            'culled_blocks': 0,
            'frame_time_ms': 0
        }
        # 缓存上一帧的可见性结果
        self.visibility_cache = {}
        # 缓存过期时间
        self.cache_lifetime = 0.5  # 秒
    
    def update(self):
        """更新视锥体"""
        if not self.enabled:
            return
            
        # 延迟初始化Frustum对象，确保camera已经存在
        if self.frustum is None:
            try:
                self.frustum = Frustum(self.camera)
            except Exception as e:
                # 如果camera还不可用，则跳过本次更新
                return
        
        # 测量更新时间
        start_time = time.time()
        
        # 动态调整更新间隔
        from ursina import application
        current_fps = getattr(application, 'fps', 30)
        
        # 根据帧率动态调整更新间隔
        if current_fps < 25:  # 低帧率
            self.update_interval = min(0.2, self.update_interval * 1.05)  # 逐渐增加间隔
            self.cache_lifetime = min(1.0, self.cache_lifetime * 1.05)  # 增加缓存生命周期
        elif current_fps > 45:  # 高帧率
            self.update_interval = max(0.05, self.update_interval * 0.95)  # 逐渐减少间隔
            self.cache_lifetime = max(0.2, self.cache_lifetime * 0.95)  # 减少缓存生命周期
            
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            try:
                self.frustum.update()
                self.last_update_time = current_time
                
                # 清理过期的缓存项
                expired_keys = []
                for entity_id, (timestamp, _) in self.visibility_cache.items():
                    if current_time - timestamp > self.cache_lifetime:
                        expired_keys.append(entity_id)
                
                for key in expired_keys:
                    del self.visibility_cache[key]
                    
            except Exception:
                # 如果更新失败，可能是camera还不可用
                pass
                
        # 记录更新耗时
        self.stats['frame_time_ms'] = (time.time() - start_time) * 1000
    
    def is_visible(self, entity, use_bounding_sphere=True):
        """判断实体是否可见"""
        if not self.enabled or self.frustum is None:
            return True  # 如果剔除功能禁用或frustum未初始化，则所有实体都可见
        
        # 检查缓存
        entity_id = id(entity)
        current_time = time.time()
        
        if entity_id in self.visibility_cache:
            timestamp, is_visible = self.visibility_cache[entity_id]
            if current_time - timestamp < self.cache_lifetime:
                return is_visible
        
        # 获取实体的位置和大小
        position = entity.position
        
        # 优化：如果启用自适应剔除，先进行距离检查
        if self.adaptive_culling and hasattr(entity, 'distance_to_player'):
            if entity.distance_to_player > self.culling_radius:
                # 超出剔除距离，直接判定为不可见
                self.visibility_cache[entity_id] = (current_time, False)
                return False
        
        # 使用包围球进行快速检测
        result = False
        if use_bounding_sphere:
            # 估算包围球半径
            radius = max(entity.scale_x, entity.scale_y, entity.scale_z) * 0.5
            result = self.frustum.is_sphere_visible(position, radius)
        else:
            # 使用轴对齐包围盒进行更精确的检测
            half_size = Vec3(entity.scale_x, entity.scale_y, entity.scale_z) * 0.5
            min_point = position - half_size
            max_point = position + half_size
            result = self.frustum.is_box_visible(min_point, max_point)
        
        # 更新缓存
        self.visibility_cache[entity_id] = (current_time, result)
        return result
    
    def filter_visible_entities(self, entities):
        """过滤出视锥体内的实体"""
        if not self.enabled or self.frustum is None:
            return entities  # 如果剔除功能禁用或frustum未初始化，则返回所有实体
        
        # 测量处理时间
        start_time = time.time()
        
        self.stats['total_blocks'] = len(entities)
        
        # 优化：批量处理，减少函数调用开销
        visible_entities = []
        for entity in entities:
            # 对于大量实体，随机跳过一些远距离实体的可见性检查
            if len(entities) > 1000 and hasattr(entity, 'distance_to_player') and entity.distance_to_player > 40:
                # 距离远的实体有50%概率跳过检查，直接判为不可见
                if random() > 0.5:
                    continue
            
            if self.is_visible(entity):
                visible_entities.append(entity)
        
        self.stats['visible_blocks'] = len(visible_entities)
        self.stats['culled_blocks'] = self.stats['total_blocks'] - self.stats['visible_blocks']
        
        # 记录处理耗时
        self.stats['frame_time_ms'] = (time.time() - start_time) * 1000
        
        return visible_entities

# 全局视锥体剔除管理器实例
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