from ursina import *
from ursina.vec3 import Vec3
from ursina.mesh import Mesh
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
        self.far_plane = 80.0  # 从100.0减少到80.0，减小视锥体范围
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
        # 只检查3个最远顶点而非全部8个
        corners = [
            Vec3(max_point.x, max_point.y, max_point.z),
            Vec3(min_point.x, max_point.y, max_point.z),
            Vec3(max_point.x, min_point.y, max_point.z)
        ]
        
        for plane_point, plane_normal in self.frustum_planes:
            # 如果所有顶点都在平面的外侧，则包围盒完全在视锥体内
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
        """检查面片是否被遮挡 - 优化的中空方块渲染"""
        # 增加距离阈值判断，远处方块不进行遮挡检查
        if face.distance_to_camera > 60:
            return True

        face_center = self._get_face_center(face)
        direction_to_camera = (camera_pos - face_center).normalized()
        
        # 根据面的类型确定检查方向
        check_direction = self._get_face_normal(face.face_type)
        
        # 背面剔除：如果面片法线与摄像机方向夹角大于90度，则剔除
        if direction_to_camera.dot(check_direction) < -0.1:  # 允许一定的容差
            return True
        
        # 相邻方块遮挡检测：检查面片外侧是否有相邻方块
        adjacent_pos = face.position + check_direction
        
        # 如果相邻位置有方块，则该面片被完全遮挡（实现中空效果）
        if adjacent_pos in self.block_positions:
            return True
            
        return False
    
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
        """添加方块并拆分为面片 - 优化的中空方块系统"""
        faces = self.split_block_to_faces(position, block_id, block_type)
        
        # 使用位置作为键而不是block_id，因为可能有多个方块使用相同ID
        position_key = (int(position.x), int(position.y), int(position.z))
        self.block_faces[position_key] = faces
        self.stats['total_faces'] += len(faces)
        
        # 更新相邻方块的可见性（中空效果的关键）
        self._update_adjacent_faces_visibility(position)
    
    def remove_block(self, position: Vec3):
        """移除方块及其面片 - 优化的中空方块系统"""
        position_key = (int(position.x), int(position.y), int(position.z))
        
        if position_key in self.block_faces:
            faces = self.block_faces[position_key]
            self.stats['total_faces'] -= len(faces)
            
            # 移除渲染实体
            for face in faces:
                face_hash = hash((face.position.x, face.position.y, face.position.z, face.face_type.value))
                if face_hash in self.render_entities:
                    destroy(self.render_entities[face_hash])
                    del self.render_entities[face_hash]
            
            del self.block_faces[position_key]
            
            # 更新相邻方块的可见性
            self._update_adjacent_faces_visibility(position)
    
    def update_culling(self, camera_pos: Vec3, camera_rotation: Vec3):
        """更新剔除计算"""
        # 更新视锥体
        self.frustum_culler.update_frustum(
            camera_pos, camera_rotation, 
            fov=90, aspect=16/9, near=0.1, far=100.0
        )
        
        # 更新遮挡剔除器的方块位置信息
        all_positions = set()
        for position_key, faces in self.block_faces.items():
            # 从位置键重建Vec3位置
            pos = Vec3(position_key[0], position_key[1], position_key[2])
            all_positions.add(pos)
        self.occlusion_culler.update_block_positions(all_positions)
        
        # 重置统计
        self.stats['visible_faces'] = 0
        self.stats['culled_faces'] = 0
        self.stats['occluded_faces'] = 0
        
        self.visible_faces.clear()
        
        # 对每个面片进行剔除测试 - 优化：先进行距离剔除，减少后续计算
        for position_key, faces in self.block_faces.items():
            # 计算区块中心到摄像机的距离
            chunk_pos = Vec3(position_key[0], position_key[1], position_key[2])
            chunk_distance = distance(chunk_pos, camera_pos)
            
            # 距离剔除 - 远处区块直接跳过详细计算
            if chunk_distance > 100.0:  # 增加距离阈值
                for face in faces:
                    face.is_visible = False
                    self.stats['culled_faces'] += 1
                continue
            
            for face in faces:
                # 计算到摄像机的距离
                face_center = self.occlusion_culler._get_face_center(face)
                face.distance_to_camera = distance(face_center, camera_pos)
                
                # 距离剔除 - 优先快速剔除
                if face.distance_to_camera > 80.0:  # 调整距离阈值
                    face.is_visible = False
                    self.stats['culled_faces'] += 1
                    continue
                
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

                # 智能面片剔除：只渲染外表面，实现中空方块效果
                # 获取面片法线
                face_normal = self.occlusion_culler._get_face_normal(face.face_type)
                # 计算摄像机到面片中心的向量
                camera_to_face = (face_center - camera_pos).normalized()
                
                # 背面剔除：如果面片背对摄像机，则剔除
                # 使用更宽松的阈值，允许侧面可见
                if face_normal.dot(-camera_to_face) < -0.2:  # 调整阈值以显示更多侧面
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
        
        # 先渲染不透明面片，再渲染透明面片
        # 分离透明和不透明面片
        opaque_batches = {}
        transparent_batches = {}
        
        for material_key, faces in self.instanced_batches.items():
            if not faces:
                continue
                
            # 检查第一个面片是否为透明材质
            is_transparent_batch = False
            if faces and len(faces) > 0:
                block_type = self._get_block_type_from_id(faces[0].block_id)
                is_transparent_batch = block_type == 'leaf'
            
            if is_transparent_batch:
                transparent_batches[material_key] = faces
            else:
                opaque_batches[material_key] = faces
        
        # 优化：限制每帧渲染的面片数量，防止卡顿
        max_faces_per_frame = 1000  # 每帧最多渲染1000个面片
        rendered_faces = 0
        
        # 1. 先渲染不透明面片（从近到远）
        for material_key, faces in opaque_batches.items():
            # 按距离排序（近到远，提高性能）
            faces.sort(key=lambda f: f.distance_to_camera, reverse=False)
            
            # 限制每批次渲染的面片数量
            faces_to_render = faces[:max_faces_per_frame - rendered_faces] if rendered_faces < max_faces_per_frame else []
            rendered_faces += len(faces_to_render)
            
            # 为每个面片创建或更新渲染实体
            for face in faces_to_render:
                self._render_single_face(face)
                self.stats['draw_calls'] += 1
        
        # 如果已经达到最大面片数，跳过透明面片渲染
        if rendered_faces >= max_faces_per_frame:
            return
        
        # 2. 再渲染透明面片（从远到近）
        for material_key, faces in transparent_batches.items():
            # 按距离排序（远到近，正确的透明度渲染）
            faces.sort(key=lambda f: f.distance_to_camera, reverse=True)
            
            # 限制每批次渲染的面片数量
            faces_to_render = faces[:max_faces_per_frame - rendered_faces] if rendered_faces < max_faces_per_frame else []
            rendered_faces += len(faces_to_render)
            
            # 为每个面片创建或更新渲染实体
            for face in faces_to_render:
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
        """调整面片实体的属性 - 优化的中空方块渲染"""
        # 创建单个面片而不是完整立方体
        # 根据面的类型创建对应的四边形面片
        vertices, triangles = self._create_face_mesh(face)
        
        # 创建自定义网格
        entity.model = Mesh(vertices=vertices, triangles=triangles, mode='triangle')
        
        # 根据方块类型设置纹理和颜色
        block_colors = {
            'grass': color.green,
            'dirt': color.brown, 
            'stone': color.gray,
            'wood': color.orange,
            'leaf': color.lime,
            'bed': color.dark_gray,
            'brick': color.red,
            'check': color.white
        }
        
        # 获取方块类型
        block_type = self._get_block_type_from_id(face.block_id)
        entity.color = block_colors.get(block_type, color.white)
        
        # 设置面片特定属性
        entity.double_sided = block_type == 'leaf'  # 树叶启用双面渲染
        
        # 设置透明度 - 树叶方块半透明
        if block_type == 'leaf':
            entity.alpha = 0.8  # 树叶设置为半透明
            entity.always_on_top = False  # 不强制在顶层渲染
        else:
            entity.alpha = 1.0  # 其他方块完全不透明
        
    def _create_face_mesh(self, face: BlockFace):
        """为单个面片创建网格数据"""
        # 面片的四个顶点（相对于面片中心）
        scale = 0.5
        
        if face.face_type == FaceType.TOP:
            vertices = [
                (-scale, 0, -scale), (scale, 0, -scale),
                (scale, 0, scale), (-scale, 0, scale)
            ]
        elif face.face_type == FaceType.BOTTOM:
            vertices = [
                (-scale, 0, scale), (scale, 0, scale),
                (scale, 0, -scale), (-scale, 0, -scale)
            ]
        elif face.face_type == FaceType.FRONT:
            vertices = [
                (-scale, -scale, 0), (-scale, scale, 0),
                (scale, scale, 0), (scale, -scale, 0)
            ]
        elif face.face_type == FaceType.BACK:
            vertices = [
                (scale, -scale, 0), (scale, scale, 0),
                (-scale, scale, 0), (-scale, -scale, 0)
            ]
        elif face.face_type == FaceType.LEFT:
            vertices = [
                (0, -scale, -scale), (0, scale, -scale),
                (0, scale, scale), (0, -scale, scale)
            ]
        elif face.face_type == FaceType.RIGHT:
            vertices = [
                (0, -scale, scale), (0, scale, scale),
                (0, scale, -scale), (0, -scale, -scale)
            ]
        
        # 三角形索引（两个三角形组成一个四边形）
        triangles = [0, 1, 2, 0, 2, 3]
        
        return vertices, triangles
        
    def _get_block_type_from_id(self, block_id: int) -> str:
        """根据方块ID获取方块类型"""
        block_types = {
            0: 'grass',
            1: 'grass', 
            2: 'stone',
            3: 'dirt',
            4: 'bed',
            5: 'wood',
            6: 'leaf',
            7: 'brick',
            8: 'check'
        }
        return block_types.get(block_id, 'default')
    
    def _update_adjacent_faces_visibility(self, position: Vec3):
        """更新相邻方块的面片可见性 - 实现中空效果的核心逻辑"""
        # 定义6个相邻方向
        adjacent_directions = [
            Vec3(0, 1, 0),   # 上
            Vec3(0, -1, 0),  # 下
            Vec3(0, 0, 1),   # 前
            Vec3(0, 0, -1),  # 后
            Vec3(-1, 0, 0),  # 左
            Vec3(1, 0, 0)    # 右
        ]
        
        # 检查每个相邻位置
        for direction in adjacent_directions:
            adjacent_pos = position + direction
            adjacent_key = (int(adjacent_pos.x), int(adjacent_pos.y), int(adjacent_pos.z))
            
            # 如果相邻位置有方块，更新其面片可见性
            if adjacent_key in self.block_faces:
                self._update_block_face_visibility(adjacent_key, adjacent_pos)
    
    def _update_block_face_visibility(self, position_key: tuple, position: Vec3):
        """更新单个方块的面片可见性"""
        if position_key not in self.block_faces:
            return
            
        faces = self.block_faces[position_key]
        
        # 检查每个面片是否应该被渲染
        for face in faces:
            # 获取面片的法线方向
            face_normal = self.occlusion_culler._get_face_normal(face.face_type)
            # 检查该方向是否有相邻方块
            adjacent_pos = position + face_normal
            adjacent_key = (int(adjacent_pos.x), int(adjacent_pos.y), int(adjacent_pos.z))
            
            # 如果相邻位置有方块，则该面片应该被隐藏（中空效果）
            face.is_occluded = adjacent_key in self.block_positions
            
            # 更新渲染实体的可见性
            face_hash = hash((face.position.x, face.position.y, face.position.z, face.face_type.value))
            if face_hash in self.render_entities:
                entity = self.render_entities[face_hash]
                entity.enabled = not face.is_occluded
    
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