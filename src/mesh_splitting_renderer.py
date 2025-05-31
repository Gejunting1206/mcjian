from ursina import *
from ursina.vec3 import Vec3
import numpy as np
from math import sqrt
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class FaceType(Enum):
    """方块面的类型枚举"""
    TOP = 0     # 上面
    BOTTOM = 1  # 下面
    FRONT = 2   # 前面
    BACK = 3    # 后面
    LEFT = 4    # 左面
    RIGHT = 5   # 右面

@dataclass
class BlockFace:
    """方块面片数据结构"""
    position: Vec3          # 面片世界坐标
    face_type: FaceType     # 面的类型
    vertices: List[Vec3]    # 4个顶点坐标
    normals: List[Vec3]     # 法线向量
    uvs: List[Tuple[float, float]]  # UV纹理坐标
    block_id: int          # 所属方块ID
    is_visible: bool = True # 是否可见
    is_occluded: bool = False # 是否被遮挡
    distance_to_camera: float = 0.0 # 到摄像机的距离

class FrustumCuller:
    """视锥体剔除器"""
    
    def __init__(self):
        self.frustum_planes = []
        self.camera_position = Vec3(0, 0, 0)
        self.camera_forward = Vec3(0, 0, -1)
        self.fov = 90
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.aspect_ratio = 16/9
    
    def update_frustum(self, camera_pos: Vec3, camera_rotation: Vec3, fov: float, aspect: float, near: float, far: float):
        """更新视锥体参数"""
        self.camera_position = camera_pos
        self.fov = fov
        self.aspect_ratio = aspect
        self.near_plane = near
        self.far_plane = far
        
        # 计算视锥体的6个平面
        self._calculate_frustum_planes(camera_pos, camera_rotation)
    
    def _calculate_frustum_planes(self, pos: Vec3, rotation: Vec3):
        """计算视锥体的6个平面方程"""
        # 计算摄像机的前、右、上向量
        forward = Vec3(0, 0, -1)
        right = Vec3(1, 0, 0)
        up = Vec3(0, 1, 0)
        
        # 应用旋转
        # 简化的旋转计算，实际应用中可能需要更精确的矩阵变换
        import math
        yaw = math.radians(rotation.y)
        pitch = math.radians(rotation.x)
        
        forward.x = math.cos(yaw) * math.cos(pitch)
        forward.y = math.sin(pitch)
        forward.z = math.sin(yaw) * math.cos(pitch)
        forward = forward.normalized()
        
        right = forward.cross(Vec3(0, 1, 0)).normalized()
        up = right.cross(forward).normalized()
        
        # 计算视锥体参数
        half_v_side = self.far_plane * math.tan(math.radians(self.fov * 0.5))
        half_h_side = half_v_side * self.aspect_ratio
        front_mult_far = forward * self.far_plane
        
        # 6个平面：近、远、左、右、上、下
        self.frustum_planes = [
            # 近平面
            (pos + forward * self.near_plane, forward),
            # 远平面
            (pos + front_mult_far, -forward),
            # 左平面
            (pos, (front_mult_far - right * half_h_side).cross(up).normalized()),
            # 右平面
            (pos, up.cross(front_mult_far + right * half_h_side).normalized()),
            # 上平面
            (pos, right.cross(front_mult_far - up * half_v_side).normalized()),
            # 下平面
            (pos, (front_mult_far + up * half_v_side).cross(right).normalized())
        ]
    
    def is_point_in_frustum(self, point: Vec3) -> bool:
        """检查点是否在视锥体内"""
        for plane_point, plane_normal in self.frustum_planes:
            # 计算点到平面的距离
            distance = (point - plane_point).dot(plane_normal)
            if distance < 0:
                return False
        return True
    
    def is_aabb_in_frustum(self, min_point: Vec3, max_point: Vec3) -> bool:
        """检查AABB包围盒是否与视锥体相交"""
        # 检查包围盒的8个顶点
        corners = [
            Vec3(min_point.x, min_point.y, min_point.z),
            Vec3(max_point.x, min_point.y, min_point.z),
            Vec3(min_point.x, max_point.y, min_point.z),
            Vec3(max_point.x, max_point.y, min_point.z),
            Vec3(min_point.x, min_point.y, max_point.z),
            Vec3(max_point.x, min_point.y, max_point.z),
            Vec3(min_point.x, max_point.y, max_point.z),
            Vec3(max_point.x, max_point.y, max_point.z)
        ]
        
        for plane_point, plane_normal in self.frustum_planes:
            # 如果所有顶点都在平面的外侧，则包围盒完全在视锥体外
            all_outside = True
            for corner in corners:
                distance = (corner - plane_point).dot(plane_normal)
                if distance >= 0:
                    all_outside = False
                    break
            if all_outside:
                return False
        return True

class OcclusionCuller:
    """遮挡剔除器"""
    
    def __init__(self):
        self.depth_buffer = {}
        self.occlusion_queries = {}
        self.block_positions = set()
    
    def update_block_positions(self, positions: Set[Vec3]):
        """更新方块位置信息"""
        self.block_positions = positions
    
    def is_face_occluded(self, face: BlockFace, camera_pos: Vec3) -> bool:
        """检查面片是否被遮挡"""
        # 简化的遮挡检测：检查面片前方是否有其他方块
        face_center = self._get_face_center(face)
        direction_to_camera = (camera_pos - face_center).normalized()
        
        # 根据面的类型确定检查方向
        check_direction = self._get_face_normal(face.face_type)
        
        # 如果面朝向与摄像机方向相反，则可能被遮挡
        if direction_to_camera.dot(check_direction) < 0:
            return True
        
        # 检查面前方是否有相邻方块
        adjacent_pos = face.position + check_direction
        return adjacent_pos in self.block_positions
    
    def _get_face_center(self, face: BlockFace) -> Vec3:
        """获取面片中心点"""
        center = Vec3(0, 0, 0)
        for vertex in face.vertices:
            center += vertex
        return center / len(face.vertices)
    
    def _get_face_normal(self, face_type: FaceType) -> Vec3:
        """获取面的法线方向"""
        normals = {
            FaceType.TOP: Vec3(0, 1, 0),
            FaceType.BOTTOM: Vec3(0, -1, 0),
            FaceType.FRONT: Vec3(0, 0, 1),
            FaceType.BACK: Vec3(0, 0, -1),
            FaceType.LEFT: Vec3(-1, 0, 0),
            FaceType.RIGHT: Vec3(1, 0, 0)
        }
        return normals[face_type]

class MeshSplittingRenderer:
    """片状拆分渲染器"""
    
    def __init__(self):
        self.block_faces: Dict[int, List[BlockFace]] = {}  # 方块ID -> 面片列表
        self.visible_faces: List[BlockFace] = []  # 当前可见的面片
        self.frustum_culler = FrustumCuller()
        self.occlusion_culler = OcclusionCuller()
        self.instanced_batches: Dict[str, List[BlockFace]] = {}  # 材质类型 -> 面片列表
        self.render_entities: Dict[int, Entity] = {}  # 面片ID -> 渲染实体
        
        # 性能统计
        self.stats = {
            'total_faces': 0,
            'visible_faces': 0,
            'culled_faces': 0,
            'occluded_faces': 0,
            'draw_calls': 0
        }
    
    def split_block_to_faces(self, position: Vec3, block_id: int, block_type: str = 'default') -> List[BlockFace]:
        """将方块拆分为6个面片"""
        faces = []
        scale = 0.5  # 方块缩放
        
        # 定义6个面的顶点（相对于方块中心）
        face_definitions = {
            FaceType.TOP: [
                Vec3(-scale, scale, -scale), Vec3(scale, scale, -scale),
                Vec3(scale, scale, scale), Vec3(-scale, scale, scale)
            ],
            FaceType.BOTTOM: [
                Vec3(-scale, -scale, scale), Vec3(scale, -scale, scale),
                Vec3(scale, -scale, -scale), Vec3(-scale, -scale, -scale)
            ],
            FaceType.FRONT: [
                Vec3(-scale, -scale, scale), Vec3(-scale, scale, scale),
                Vec3(scale, scale, scale), Vec3(scale, -scale, scale)
            ],
            FaceType.BACK: [
                Vec3(scale, -scale, -scale), Vec3(scale, scale, -scale),
                Vec3(-scale, scale, -scale), Vec3(-scale, -scale, -scale)
            ],
            FaceType.LEFT: [
                Vec3(-scale, -scale, -scale), Vec3(-scale, scale, -scale),
                Vec3(-scale, scale, scale), Vec3(-scale, -scale, scale)
            ],
            FaceType.RIGHT: [
                Vec3(scale, -scale, scale), Vec3(scale, scale, scale),
                Vec3(scale, scale, -scale), Vec3(scale, -scale, -scale)
            ]
        }
        
        # UV坐标（标准四边形）
        standard_uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        for face_type, vertices in face_definitions.items():
            # 转换到世界坐标
            world_vertices = [position + vertex for vertex in vertices]
            
            # 计算法线
            normal = self.occlusion_culler._get_face_normal(face_type)
            normals = [normal] * 4
            
            face = BlockFace(
                position=position,
                face_type=face_type,
                vertices=world_vertices,
                normals=normals,
                uvs=standard_uvs,
                block_id=block_id
            )
            
            faces.append(face)
        
        return faces
    
    def add_block(self, position: Vec3, block_id: int, block_type: str = 'default'):
        """添加方块并拆分为面片"""
        faces = self.split_block_to_faces(position, block_id, block_type)
        self.block_faces[block_id] = faces
        self.stats['total_faces'] += len(faces)
    
    def remove_block(self, block_id: int):
        """移除方块及其面片"""
        if block_id in self.block_faces:
            faces = self.block_faces[block_id]
            self.stats['total_faces'] -= len(faces)
            
            # 移除渲染实体
            for face in faces:
                face_hash = hash((face.position.x, face.position.y, face.position.z, face.face_type.value))
                if face_hash in self.render_entities:
                    destroy(self.render_entities[face_hash])
                    del self.render_entities[face_hash]
            
            del self.block_faces[block_id]
    
    def update_culling(self, camera_pos: Vec3, camera_rotation: Vec3):
        """更新剔除计算"""
        # 更新视锥体
        self.frustum_culler.update_frustum(
            camera_pos, camera_rotation, 
            fov=90, aspect=16/9, near=0.1, far=100.0
        )
        
        # 更新遮挡剔除器的方块位置信息
        all_positions = set()
        for faces in self.block_faces.values():
            for face in faces:
                all_positions.add(face.position)
        self.occlusion_culler.update_block_positions(all_positions)
        
        # 重置统计
        self.stats['visible_faces'] = 0
        self.stats['culled_faces'] = 0
        self.stats['occluded_faces'] = 0
        
        self.visible_faces.clear()
        
        # 对每个面片进行剔除测试
        for block_id, faces in self.block_faces.items():
            for face in faces:
                # 计算到摄像机的距离
                face_center = self.occlusion_culler._get_face_center(face)
                face.distance_to_camera = distance(face_center, camera_pos)
                
                # 视锥体剔除
                face_min = Vec3(
                    min(v.x for v in face.vertices),
                    min(v.y for v in face.vertices),
                    min(v.z for v in face.vertices)
                )
                face_max = Vec3(
                    max(v.x for v in face.vertices),
                    max(v.y for v in face.vertices),
                    max(v.z for v in face.vertices)
                )
                
                if not self.frustum_culler.is_aabb_in_frustum(face_min, face_max):
                    face.is_visible = False
                    self.stats['culled_faces'] += 1
                    continue
                
                # 遮挡剔除
                if self.occlusion_culler.is_face_occluded(face, camera_pos):
                    face.is_occluded = True
                    self.stats['occluded_faces'] += 1
                    continue

                # 新增：根据面片法线和摄像机方向进行剔除
                # 获取面片法线
                face_normal = self.occlusion_culler._get_face_normal(face.face_type)
                # 计算摄像机到面片中心的向量
                camera_to_face = (face_center - camera_pos).normalized()
                
                # 如果面片法线与摄像机到面片中心的向量点积小于0，说明面片背对摄像机，进行剔除
                # 这里的阈值可以根据“两三面”的需求进行调整，目前是严格剔除背对面
                if face_normal.dot(camera_to_face) > 0.1: # 稍微大于0，避免浮点误差，并允许一定角度的侧面
                    face.is_visible = False
                    self.stats['culled_faces'] += 1
                    continue
                
                # 面片可见
                face.is_visible = True
                face.is_occluded = False
                self.visible_faces.append(face)
                self.stats['visible_faces'] += 1
    
    def create_instanced_batches(self):
        """创建实例化渲染批次"""
        self.instanced_batches.clear()
        
        # 按材质类型分组可见面片
        for face in self.visible_faces:
            # 简化：所有面片使用相同材质
            material_key = f"block_{face.face_type.name.lower()}"
            
            if material_key not in self.instanced_batches:
                self.instanced_batches[material_key] = []
            
            self.instanced_batches[material_key].append(face)
    
    def render_faces(self):
        """渲染可见面片"""
        self.stats['draw_calls'] = 0
        
        # 创建实例化批次
        self.create_instanced_batches()
        
        # 渲染每个批次
        for material_key, faces in self.instanced_batches.items():
            if not faces:
                continue
            
            # 按距离排序（远到近，用于透明度渲染）
            faces.sort(key=lambda f: f.distance_to_camera, reverse=True)
            
            # 为每个面片创建或更新渲染实体
            for face in faces:
                self._render_single_face(face)
                self.stats['draw_calls'] += 1
    
    def _render_single_face(self, face: BlockFace):
        """渲染单个面片"""
        face_hash = hash((face.position.x, face.position.y, face.position.z, face.face_type.value))
        
        if face_hash not in self.render_entities:
            # 创建新的渲染实体
            entity = Entity(
                model='cube',
                position=face.position,
                scale=0.5,
                color=color.white
            )
            
            # 根据面的类型调整实体
            self._adjust_face_entity(entity, face)
            
            self.render_entities[face_hash] = entity
        else:
            # 更新现有实体
            entity = self.render_entities[face_hash]
            entity.position = face.position
            entity.enabled = face.is_visible and not face.is_occluded
    
    def _adjust_face_entity(self, entity: Entity, face: BlockFace):
        """调整面片实体的属性"""
        # 根据面的类型设置不同的颜色（用于调试）
        face_colors = {
            FaceType.TOP: color.green,
            FaceType.BOTTOM: color.brown,
            FaceType.FRONT: color.blue,
            FaceType.BACK: color.red,
            FaceType.LEFT: color.yellow,
            FaceType.RIGHT: color.orange
        }
        
        entity.color = face_colors.get(face.face_type, color.white)
        
        # 可以在这里添加更多面片特定的调整
        # 例如：纹理映射、材质属性等
    
    def get_performance_stats(self) -> Dict[str, int]:
        """获取性能统计信息"""
        return self.stats.copy()
    
    def cleanup(self):
        """清理资源"""
        # 销毁所有渲染实体
        for entity in self.render_entities.values():
            destroy(entity)
        
        self.render_entities.clear()
        self.block_faces.clear()
        self.visible_faces.clear()
        self.instanced_batches.clear()

# 全局渲染器实例
mesh_renderer = MeshSplittingRenderer()