# GPU加速视锥体剔除模块
# 使用GPU并行计算加速视锥体剔除过程

from ursina import *
import numpy as np
import time
from collections import defaultdict, deque
import math

class GPUFrustumCulling:
    """GPU加速视锥体剔除 - 使用GPU并行计算加速剔除过程"""
    
    def __init__(self):
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒）
        
        # 视锥体参数
        self.frustum = None
        self.frustum_planes = []  # 视锥体的6个平面
        self.frustum_corners = []  # 视锥体的8个角点
        
        # 空间分区
        self.use_spatial_hash = True
        self.spatial_hash_size = 16  # 空间哈希网格大小
        self.spatial_grid = defaultdict(list)  # 空间哈希网格
        
        # 八叉树参数
        self.use_octree = True
        self.octree_max_depth = 5
        self.octree_min_size = 4
        self.octree_root = None
        self.octree_rebuild_interval = 1.0  # 八叉树重建间隔
        self.last_octree_rebuild = 0
        
        # 视锥体缓存
        self.use_frustum_cache = True
        self.frustum_cache = {}
        self.frustum_cache_lifetime = 0.1  # 缓存生命周期（秒）
        
        # 批处理参数
        self.batch_size = 64  # 每批处理的实体数量
        
        # 统计信息
        self.stats = {
            'total_entities': 0,
            'visible_entities': 0,
            'culled_entities': 0,
            'culling_ratio': 0.0,
            'culling_time_ms': 0.0,
            'spatial_cells': 0,
            'octree_nodes': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 初始化着色器
        self._init_culling_shader()
    
    def _init_culling_shader(self):
        """初始化用于GPU剔除的着色器"""
        # 计算着色器，用于批量视锥体测试
        self.culling_shader = Shader(
            compute='''
            #version 430
            
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            
            // 输入数据
            layout(std430, binding = 0) buffer EntityPositions {
                vec4 positions[];
            };
            
            layout(std430, binding = 1) buffer EntitySizes {
                vec4 sizes[];
            };
            
            // 输出结果
            layout(std430, binding = 2) buffer CullingResults {
                int results[];
            };
            
            // 视锥体平面
            uniform vec4 frustum_planes[6];
            
            // 判断点是否在平面内侧
            bool isPointInside(vec3 point, vec4 plane) {
                return dot(vec3(plane), point) + plane.w >= 0.0;
            }
            
            // 判断包围盒是否与平面相交
            bool isBoxIntersectingPlane(vec3 center, vec3 extents, vec4 plane) {
                float r = extents.x * abs(plane.x) + extents.y * abs(plane.y) + extents.z * abs(plane.z);
                float d = dot(vec3(plane), center) + plane.w;
                return d >= -r;
            }
            
            void main() {
                uint id = gl_GlobalInvocationID.x;
                
                // 确保不超出数组范围
                if (id >= positions.length()) {
                    return;
                }
                
                // 获取实体位置和大小
                vec3 position = positions[id].xyz;
                vec3 size = sizes[id].xyz;
                
                // 默认可见
                results[id] = 1;
                
                // 对每个视锥体平面进行测试
                for (int i = 0; i < 6; i++) {
                    if (!isBoxIntersectingPlane(position, size * 0.5, frustum_planes[i])) {
                        // 在平面外侧，不可见
                        results[id] = 0;
                        break;
                    }
                }
            }
            '''
        )
        
        # 创建缓冲区
        self.position_buffer = None
        self.size_buffer = None
        self.result_buffer = None
    
    def update_frustum(self, camera):
        """更新视锥体平面"""
        if not camera:
            return
        
        # 获取相机的视图矩阵和投影矩阵
        view_matrix = camera.view_matrix
        projection_matrix = camera.projection_matrix
        
        # 计算视图投影矩阵
        vp_matrix = projection_matrix * view_matrix
        
        # 提取视锥体平面
        self.frustum_planes = self._extract_frustum_planes(vp_matrix)
        
        # 计算视锥体角点（用于八叉树优化）
        self.frustum_corners = self._calculate_frustum_corners(vp_matrix)
        
        # 更新完成后，清除缓存
        if self.use_frustum_cache:
            self.frustum_cache.clear()
    
    def _extract_frustum_planes(self, vp_matrix):
        """从视图投影矩阵中提取视锥体平面"""
        # 提取矩阵元素
        m = vp_matrix.flatten()
        
        # 计算6个平面（左、右、下、上、近、远）
        planes = [
            Vec4(m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]).normalized(),  # 左
            Vec4(m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]).normalized(),  # 右
            Vec4(m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]).normalized(),  # 下
            Vec4(m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]).normalized(),  # 上
            Vec4(m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]).normalized(),  # 近
            Vec4(m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]).normalized()   # 远
        ]
        
        return planes
    
    def _calculate_frustum_corners(self, vp_matrix):
        """计算视锥体的8个角点"""
        # 逆矩阵
        inv_vp = vp_matrix.get_inverse()
        
        # 定义NDC空间中的8个角点
        ndc_corners = [
            Vec4(-1, -1, -1, 1),  # 近平面左下
            Vec4(1, -1, -1, 1),   # 近平面右下
            Vec4(1, 1, -1, 1),    # 近平面右上
            Vec4(-1, 1, -1, 1),   # 近平面左上
            Vec4(-1, -1, 1, 1),   # 远平面左下
            Vec4(1, -1, 1, 1),    # 远平面右下
            Vec4(1, 1, 1, 1),     # 远平面右上
            Vec4(-1, 1, 1, 1)     # 远平面左上
        ]
        
        # 转换到世界空间
        world_corners = []
        for ndc in ndc_corners:
            world_point = inv_vp * ndc
            world_point /= world_point.w
            world_corners.append(Vec3(world_point.x, world_point.y, world_point.z))
        
        return world_corners
    
    def update_spatial_structures(self, entities):
        """更新空间数据结构（空间哈希和八叉树）"""
        current_time = time.time()
        
        # 更新空间哈希
        if self.use_spatial_hash:
            self._update_spatial_hash(entities)
        
        # 定期重建八叉树
        if self.use_octree and current_time - self.last_octree_rebuild > self.octree_rebuild_interval:
            self._rebuild_octree(entities)
            self.last_octree_rebuild = current_time
    
    def _update_spatial_hash(self, entities):
        """更新空间哈希网格"""
        # 清空现有网格
        self.spatial_grid.clear()
        
        # 将实体添加到网格
        for entity in entities:
            if hasattr(entity, 'position'):
                # 计算网格坐标
                grid_x = int(entity.position.x / self.spatial_hash_size)
                grid_y = int(entity.position.y / self.spatial_hash_size)
                grid_z = int(entity.position.z / self.spatial_hash_size)
                
                # 添加到网格
                grid_key = (grid_x, grid_y, grid_z)
                self.spatial_grid[grid_key].append(entity)
        
        # 更新统计信息
        self.stats['spatial_cells'] = len(self.spatial_grid)
    
    def _rebuild_octree(self, entities):
        """重建八叉树"""
        # 确定场景边界
        min_bounds = Vec3(float('inf'), float('inf'), float('inf'))
        max_bounds = Vec3(float('-inf'), float('-inf'), float('-inf'))
        
        for entity in entities:
            if hasattr(entity, 'position'):
                min_bounds.x = min(min_bounds.x, entity.position.x - 1)
                min_bounds.y = min(min_bounds.y, entity.position.y - 1)
                min_bounds.z = min(min_bounds.z, entity.position.z - 1)
                
                max_bounds.x = max(max_bounds.x, entity.position.x + 1)
                max_bounds.y = max(max_bounds.y, entity.position.y + 1)
                max_bounds.z = max(max_bounds.z, entity.position.z + 1)
        
        # 创建根节点
        center = (min_bounds + max_bounds) * 0.5
        half_size = (max_bounds - min_bounds) * 0.5
        
        # 构建八叉树
        self.octree_root = OctreeNode(center, half_size, 0, self.octree_max_depth, self.octree_min_size)
        
        # 插入实体
        for entity in entities:
            if hasattr(entity, 'position'):
                self.octree_root.insert(entity)
        
        # 更新统计信息
        self.stats['octree_nodes'] = self.octree_root.count_nodes()
    
    def filter_visible_entities(self, entities, camera_position=None):
        """过滤出视锥体内的实体"""
        if not self.enabled or not self.frustum_planes:
            return entities
        
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if self.use_frustum_cache and camera_position:
            # 使用相机位置和朝向作为缓存键
            if hasattr(camera, 'rotation'):
                cache_key = (tuple(camera_position), tuple(camera.rotation))
            else:
                cache_key = tuple(camera_position)
            
            # 检查缓存是否有效
            if cache_key in self.frustum_cache:
                cache_entry = self.frustum_cache[cache_key]
                if time.time() - cache_entry['time'] < self.frustum_cache_lifetime:
                    self.stats['cache_hits'] += 1
                    return cache_entry['visible_entities']
            
            self.stats['cache_misses'] += 1
        
        # 根据使用的空间结构选择不同的剔除策略
        if self.use_spatial_hash and self.spatial_grid:
            visible_entities = self._filter_using_spatial_hash(camera_position)
        elif self.use_octree and self.octree_root:
            visible_entities = self._filter_using_octree()
        else:
            visible_entities = self._filter_using_gpu(entities)
        
        # 更新统计信息
        self.stats['total_entities'] = len(entities)
        self.stats['visible_entities'] = len(visible_entities)
        self.stats['culled_entities'] = len(entities) - len(visible_entities)
        
        if len(entities) > 0:
            self.stats['culling_ratio'] = self.stats['culled_entities'] / len(entities)
        
        self.stats['culling_time_ms'] = (time.time() - start_time) * 1000
        
        # 更新缓存
        if self.use_frustum_cache and cache_key:
            self.frustum_cache[cache_key] = {
                'time': time.time(),
                'visible_entities': visible_entities
            }
            
            # 清理过期缓存
            self._clean_frustum_cache()
        
        return visible_entities
    
    def _filter_using_gpu(self, entities):
        """使用GPU并行计算进行视锥体剔除"""
        # 准备数据
        positions = []
        sizes = []
        
        for entity in entities:
            if hasattr(entity, 'position'):
                # 获取位置
                pos = entity.position
                
                # 获取大小（包围盒）
                size = Vec3(1, 1, 1)  # 默认大小
                if hasattr(entity, 'scale'):
                    size = entity.scale
                
                positions.append(Vec4(pos.x, pos.y, pos.z, 1.0))
                sizes.append(Vec4(size.x, size.y, size.z, 0.0))
        
        # 如果没有实体，直接返回空列表
        if not positions:
            return []
        
        # 创建或更新缓冲区
        if not self.position_buffer or len(positions) != len(self.position_buffer):
            self.position_buffer = np.array(positions, dtype=np.float32)
            self.size_buffer = np.array(sizes, dtype=np.float32)
            self.result_buffer = np.zeros(len(positions), dtype=np.int32)
        else:
            # 更新现有缓冲区
            for i, pos in enumerate(positions):
                self.position_buffer[i] = [pos.x, pos.y, pos.z, 1.0]
                self.size_buffer[i] = [sizes[i].x, sizes[i].y, sizes[i].z, 0.0]
            self.result_buffer.fill(0)
        
        # 设置着色器参数
        shader = self.culling_shader
        shader.set_shader_input('frustum_planes', self.frustum_planes)
        shader.set_shader_input('positions', self.position_buffer)
        shader.set_shader_input('sizes', self.size_buffer)
        shader.set_shader_input('results', self.result_buffer)
        
        # 计算分派组数
        group_count = (len(positions) + 63) // 64  # 每组64个实体
        
        # 执行计算着色器
        shader.dispatch(group_count, 1, 1)
        
        # 读取结果
        visible_indices = np.where(self.result_buffer == 1)[0]
        visible_entities = [entities[i] for i in visible_indices]
        
        return visible_entities
    
    def _filter_using_spatial_hash(self, camera_position):
        """使用空间哈希网格进行视锥体剔除"""
        visible_entities = []
        
        # 确定需要检查的网格单元
        cells_to_check = self._get_cells_intersecting_frustum()
        
        # 对每个相关的网格单元进行处理
        for cell_key in cells_to_check:
            if cell_key in self.spatial_grid:
                entities_in_cell = self.spatial_grid[cell_key]
                
                # 批量处理实体
                for i in range(0, len(entities_in_cell), self.batch_size):
                    batch = entities_in_cell[i:i+self.batch_size]
                    
                    # 使用GPU进行批量剔除测试
                    visible_batch = self._filter_using_gpu(batch)
                    visible_entities.extend(visible_batch)
        
        return visible_entities
    
    def _get_cells_intersecting_frustum(self):
        """获取与视锥体相交的网格单元"""
        # 使用视锥体角点确定边界
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for corner in self.frustum_corners:
            min_x = min(min_x, corner.x)
            min_y = min(min_y, corner.y)
            min_z = min(min_z, corner.z)
            
            max_x = max(max_x, corner.x)
            max_y = max(max_y, corner.y)
            max_z = max(max_z, corner.z)
        
        # 转换为网格坐标
        grid_min_x = int(min_x / self.spatial_hash_size) - 1
        grid_min_y = int(min_y / self.spatial_hash_size) - 1
        grid_min_z = int(min_z / self.spatial_hash_size) - 1
        
        grid_max_x = int(max_x / self.spatial_hash_size) + 1
        grid_max_y = int(max_y / self.spatial_hash_size) + 1
        grid_max_z = int(max_z / self.spatial_hash_size) + 1
        
        # 生成网格单元列表
        cells = []
        for x in range(grid_min_x, grid_max_x + 1):
            for y in range(grid_min_y, grid_max_y + 1):
                for z in range(grid_min_z, grid_max_z + 1):
                    cells.append((x, y, z))
        
        return cells
    
    def _filter_using_octree(self):
        """使用八叉树进行视锥体剔除"""
        if not self.octree_root:
            return []
        
        # 收集与视锥体相交的节点中的实体
        visible_entities = []
        self.octree_root.get_visible_entities(self.frustum_planes, visible_entities)
        
        return visible_entities
    
    def _clean_frustum_cache(self):
        """清理过期的视锥体缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.frustum_cache.items():
            if current_time - entry['time'] > self.frustum_cache_lifetime:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.frustum_cache[key]
    
    def update(self, camera=None):
        """更新视锥体剔除系统"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # 降低更新频率
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 更新视锥体
        if camera:
            self.update_frustum(camera)
    
    def toggle(self):
        """切换视锥体剔除"""
        self.enabled = not self.enabled
        return self.enabled
    
    def set_update_interval(self, interval):
        """设置更新间隔"""
        self.update_interval = max(0.01, interval)
    
    def set_spatial_hash_size(self, size):
        """设置空间哈希网格大小"""
        self.spatial_hash_size = max(4, size)
        # 清空网格，强制下次更新重建
        self.spatial_grid.clear()
    
    def set_octree_depth(self, depth):
        """设置八叉树最大深度"""
        self.octree_max_depth = max(1, min(8, depth))
        # 强制下次更新重建八叉树
        self.octree_root = None
        self.last_octree_rebuild = 0
    
    def set_frustum_cache_lifetime(self, lifetime):
        """设置视锥体缓存生命周期"""
        self.frustum_cache_lifetime = max(0.01, lifetime)
        # 清空缓存
        self.frustum_cache.clear()

class OctreeNode:
    """八叉树节点"""
    
    def __init__(self, center, half_size, depth, max_depth, min_size):
        self.center = center
        self.half_size = half_size
        self.depth = depth
        self.max_depth = max_depth
        self.min_size = min_size
        
        self.entities = []
        self.children = [None] * 8  # 8个子节点
        self.is_leaf = True
    
    def insert(self, entity):
        """插入实体到八叉树"""
        # 检查实体是否在节点范围内
        if not self._contains(entity.position):
            return False
        
        # 如果是叶节点且深度未达到最大值，且尺寸足够大，考虑细分
        if self.is_leaf and self.depth < self.max_depth and self.half_size.x > self.min_size:
            # 如果实体数量超过阈值，细分节点
            if len(self.entities) >= 8:
                self._split()
        
        # 如果已细分，尝试将实体插入到子节点
        if not self.is_leaf:
            # 确定实体所在的子节点索引
            index = self._get_child_index(entity.position)
            
            # 插入到子节点
            if self.children[index].insert(entity):
                return True
        
        # 如果是叶节点或无法插入到子节点，存储在当前节点
        self.entities.append(entity)
        return True
    
    def _split(self):
        """细分节点为8个子节点"""
        self.is_leaf = False
        new_half_size = self.half_size * 0.5
        new_depth = self.depth + 1
        
        # 创建8个子节点
        for i in range(8):
            # 确定子节点中心
            offset = Vec3(
                ((i & 1) * 2 - 1) * new_half_size.x,
                ((i & 2) - 1) * new_half_size.y,
                ((i & 4) / 2 - 1) * new_half_size.z
            )
            
            child_center = self.center + offset
            
            # 创建子节点
            self.children[i] = OctreeNode(child_center, new_half_size, new_depth, self.max_depth, self.min_size)
        
        # 重新分配现有实体
        entities_to_redistribute = self.entities.copy()
        self.entities.clear()
        
        for entity in entities_to_redistribute:
            self.insert(entity)
    
    def _get_child_index(self, position):
        """确定位置所在的子节点索引"""
        index = 0
        if position.x >= self.center.x: index |= 1
        if position.y >= self.center.y: index |= 2
        if position.z >= self.center.z: index |= 4
        return index
    
    def _contains(self, position):
        """检查位置是否在节点范围内"""
        return (
            abs(position.x - self.center.x) <= self.half_size.x and
            abs(position.y - self.center.y) <= self.half_size.y and
            abs(position.z - self.center.z) <= self.half_size.z
        )
    
    def get_visible_entities(self, frustum_planes, result_list):
        """获取视锥体内的实体"""
        # 检查节点是否与视锥体相交
        if not self._intersects_frustum(frustum_planes):
            return
        
        # 如果是叶节点，添加所有实体
        if self.is_leaf:
            result_list.extend(self.entities)
        else:
            # 递归检查子节点
            for child in self.children:
                if child:
                    child.get_visible_entities(frustum_planes, result_list)
            
            # 添加当前节点的实体
            result_list.extend(self.entities)
    
    def _intersects_frustum(self, frustum_planes):
        """检查节点是否与视锥体相交"""
        # 对每个平面进行测试
        for plane in frustum_planes:
            # 计算节点到平面的距离
            r = self.half_size.x * abs(plane.x) + self.half_size.y * abs(plane.y) + self.half_size.z * abs(plane.z)
            d = plane.x * self.center.x + plane.y * self.center.y + plane.z * self.center.z + plane.w
            
            # 如果节点完全在平面外侧，则不相交
            if d < -r:
                return False
        
        return True
    
    def count_nodes(self):
        """计算八叉树中的节点数量"""
        count = 1  # 当前节点
        
        if not self.is_leaf:
            for child in self.children:
                if child:
                    count += child.count_nodes()
        
        return count

# 创建全局实例
gpu_frustum_culling = GPUFrustumCulling()