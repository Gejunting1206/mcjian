from ursina import *
import time
import math
from collections import defaultdict

class SpatialHashGrid:
    """空间哈希网格，用于快速查找指定区域内的对象，针对16区块渲染距离优化"""
    
    def __init__(self, cell_size=64):
        self.cell_size = cell_size  # 从32增加到64，减少网格数量
        self.grid = defaultdict(set)  # 使用defaultdict避免频繁检查键是否存在
        self.object_to_cells = {}  # 对象到网格单元的映射
        self.stats = {'queries': 0, 'objects_processed': 0, 'cells_checked': 0}
        self.enabled = True
        self.max_objects_per_cell = 100  # 每个网格最大对象数量
        self.use_quadtree_fallback = True  # 当网格对象过多时使用四叉树
        self.quadtrees = {}  # 存储过载网格的四叉树
        self.last_cleanup_time = 0
        self.cleanup_interval = 10.0  # 清理间隔（秒）
    
    def _hash_position(self, position):
        """将3D位置哈希到网格坐标"""
        return (
            int(position.x // self.cell_size),
            int(position.y // self.cell_size), 
            int(position.z // self.cell_size)
        )
    
    def _get_cell_coords(self, position):
        """获取位置对应的网格坐标（与_hash_position功能相同，保持兼容性）"""
        return self._hash_position(position)
    
    def insert(self, obj, position):
        """插入对象到空间哈希网格"""
        if not self.enabled or not hasattr(position, 'x'):
            return
            
        cell = self._hash_position(position)
        
        # 检查是否需要使用四叉树
        if self.use_quadtree_fallback and len(self.grid[cell]) >= self.max_objects_per_cell:
            if cell not in self.quadtrees:
                # 创建新的四叉树
                min_x = cell[0] * self.cell_size
                min_y = cell[1] * self.cell_size
                min_z = cell[2] * self.cell_size
                self.quadtrees[cell] = QuadTree(
                    Rect(min_x, min_y, self.cell_size, self.cell_size),
                    min_z, min_z + self.cell_size
                )
                # 将现有对象添加到四叉树
                for existing_obj in self.grid[cell]:
                    if existing_obj in self.object_to_cells and hasattr(existing_obj, 'position') and existing_obj.enabled:
                        self.quadtrees[cell].insert(existing_obj, existing_obj.position)
            
            # 添加到四叉树
            self.quadtrees[cell].insert(obj, position)
        
        # 添加到网格
        self.grid[cell].add(obj)
        self.object_to_cells[obj] = cell
    
    def remove(self, obj):
        """从空间哈希网格移除对象"""
        if not self.enabled or obj not in self.object_to_cells:
            return
            
        cell = self.object_to_cells[obj]
        
        # 从网格中移除
        if cell in self.grid:
            self.grid[cell].discard(obj)
            if not self.grid[cell]:  # 如果网格单元为空，删除它
                del self.grid[cell]
        
        # 从四叉树中移除
        if cell in self.quadtrees:
            self.quadtrees[cell].remove(obj)
            # 如果四叉树为空，删除它
            if self.quadtrees[cell].is_empty():
                del self.quadtrees[cell]
        
        # 从映射中移除
        del self.object_to_cells[obj]
    
    def update(self, obj, position):
        """更新对象位置"""
        if not self.enabled:
            return
            
        old_cell = self.object_to_cells.get(obj)
        new_cell = self._hash_position(position)
        
        # 如果单元格没有变化，不需要更新
        if old_cell == new_cell:
            return
        
        # 移除旧位置
        self.remove(obj)
        
        # 添加到新位置
        self.insert(obj, position)
    
    def query_nearby(self, position, radius=1):
        """查询附近的对象，针对16区块渲染距离优化"""
        if not self.enabled or not hasattr(position, 'x'):
            return set()
        
        self.stats['queries'] += 1
        center_cell = self._hash_position(position)
        nearby_objects = set()
        
        # 计算网格半径 - 根据实际距离计算需要检查的网格数量
        grid_radius = max(1, int(radius // self.cell_size) + 1)
        
        # 优化：首先检查中心单元格
        if center_cell in self.grid:
            nearby_objects.update(self.grid[center_cell])
            
            # 检查中心单元格的四叉树
            if center_cell in self.quadtrees:
                # 计算查询范围
                query_rect = Rect(
                    position.x - radius, position.y - radius,
                    radius * 2, radius * 2
                )
                # 从四叉树中查询
                quad_objects = self.quadtrees[center_cell].query(query_rect, position.z - radius, position.z + radius)
                nearby_objects.update(quad_objects)
        
        # 检查周围的网格单元
        cells_checked = 1  # 已经检查了中心单元格
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                for dz in range(-grid_radius, grid_radius + 1):
                    # 跳过中心单元格，因为已经检查过了
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    # 优化：跳过对角线上的远端单元格
                    if abs(dx) == grid_radius and abs(dy) == grid_radius and abs(dz) == grid_radius:
                        continue
                    
                    cell = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                    cells_checked += 1
                    
                    if cell in self.grid:
                        nearby_objects.update(self.grid[cell])
                        
                        # 检查该单元格的四叉树
                        if cell in self.quadtrees:
                            # 计算查询范围
                            query_rect = Rect(
                                position.x - radius, position.y - radius,
                                radius * 2, radius * 2
                            )
                            # 从四叉树中查询
                            quad_objects = self.quadtrees[cell].query(query_rect, position.z - radius, position.z + radius)
                            nearby_objects.update(quad_objects)
        
        # 更新统计信息
        self.stats['cells_checked'] += cells_checked
        self.stats['objects_processed'] += len(nearby_objects)
        
        # 定期清理未使用的单元格
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_unused_cells(position, grid_radius * 3)
            self.last_cleanup_time = current_time
        
        return nearby_objects
    
    def query_radius(self, position, radius):
        """查询指定半径内的对象"""
        if not self.enabled:
            return set()
            
        nearby_objects = self.query_nearby(position, radius)
        result = set()
        
        # 过滤出在实际半径内的对象
        for obj in nearby_objects:
            if hasattr(obj, 'position'):
                # 计算实际距离
                dx = obj.position.x - position.x
                dy = obj.position.y - position.y
                dz = obj.position.z - position.z
                distance_squared = dx*dx + dy*dy + dz*dz
                
                if distance_squared <= radius*radius:
                    result.add(obj)
        
        return result
    
    def _cleanup_unused_cells(self, position, max_distance):
        """清理远离指定位置的未使用网格单元以节省内存"""
        if not self.grid:
            return
            
        center_cell = self._hash_position(position)
        cells_to_remove = []
        
        # 找出距离过远的单元格
        for cell_coords in list(self.grid.keys()):
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
                # 清理对象到单元格的映射
                for obj in self.grid[cell_coords]:
                    if obj in self.object_to_cells and obj.enabled:
                        del self.object_to_cells[obj]
                # 删除单元格
                del self.grid[cell_coords]
            
            # 清理四叉树
            if cell_coords in self.quadtrees:
                del self.quadtrees[cell_coords]
    
    def clear(self):
        """清空空间哈希网格"""
        self.grid.clear()
        self.object_to_cells.clear()
        self.quadtrees.clear()
        self.stats = {'queries': 0, 'objects_processed': 0, 'cells_checked': 0}


class Rect:
    """矩形区域，用于四叉树"""
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def contains(self, point):
        """检查点是否在矩形内"""
        return (
            point.x >= self.x and
            point.x < self.x + self.width and
            point.y >= self.y and
            point.y < self.y + self.height
        )
    
    def intersects(self, other):
        """检查是否与另一个矩形相交"""
        return not (
            other.x > self.x + self.width or
            other.x + other.width < self.x or
            other.y > self.y + self.height or
            other.y + other.height < self.y
        )


class QuadTree:
    """四叉树，用于高效空间查询"""
    
    def __init__(self, boundary, min_z, max_z, capacity=10, max_depth=4, depth=0):
        self.boundary = boundary
        self.min_z = min_z
        self.max_z = max_z
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.objects = []
        self.divided = False
        self.children = []
    
    def insert(self, obj, position):
        """插入对象到四叉树"""
        # 检查位置是否在边界内
        if not self.boundary.contains(position) or position.z < self.min_z or position.z >= self.max_z:
            return False
        
        # 如果还有空间且未分割，直接添加
        if len(self.objects) < self.capacity and not self.divided:
            self.objects.append(obj)
            return True
        
        # 如果未分割且深度未达到最大值，进行分割
        if not self.divided and self.depth < self.max_depth:
            self._subdivide()
        
        # 如果已分割，尝试添加到子节点
        if self.divided:
            for child in self.children:
                if child.insert(obj, position):
                    return True
        
        # 如果无法添加到子节点或已达到最大深度，添加到当前节点
        self.objects.append(obj)
        return True
    
    def _subdivide(self):
        """将四叉树分割为四个子节点"""
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.width / 2
        h = self.boundary.height / 2
        z_mid = (self.min_z + self.max_z) / 2
        
        # 创建四个子节点
        self.children = [
            QuadTree(Rect(x, y, w, h), self.min_z, z_mid, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x + w, y, w, h), self.min_z, z_mid, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x, y + h, w, h), self.min_z, z_mid, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x + w, y + h, w, h), self.min_z, z_mid, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x, y, w, h), z_mid, self.max_z, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x + w, y, w, h), z_mid, self.max_z, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x, y + h, w, h), z_mid, self.max_z, self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(x + w, y + h, w, h), z_mid, self.max_z, self.capacity, self.max_depth, self.depth + 1)
        ]
        
        self.divided = True
        
        # 将现有对象重新分配到子节点
        objects_to_redistribute = self.objects.copy()
        self.objects.clear()
        
        for obj in objects_to_redistribute:
            if hasattr(obj, 'position'):
                inserted = False
                for child in self.children:
                    if child.insert(obj, obj.position):
                        inserted = True
                        break
                
                # 如果无法插入到任何子节点，保留在当前节点
                if not inserted:
                    self.objects.append(obj)
    
    def query(self, range_rect, min_z, max_z):
        """查询指定范围内的对象"""
        result = set()
        
        # 检查范围是否与边界相交
        if not self.boundary.intersects(range_rect) or max_z < self.min_z or min_z >= self.max_z:
            return result
        
        # 添加当前节点中的对象
        for obj in self.objects:
            if hasattr(obj, 'position'):
                if range_rect.contains(obj.position) and obj.position.z >= min_z and obj.position.z < max_z:
                    result.add(obj)
        
        # 如果已分割，查询子节点
        if self.divided:
            for child in self.children:
                result.update(child.query(range_rect, min_z, max_z))
        
        return result
    
    def remove(self, obj):
        """从四叉树中移除对象"""
        # 检查当前节点
        if obj in self.objects:
            self.objects.remove(obj)
            return True
        
        # 检查子节点
        if self.divided:
            for child in self.children:
                if child.remove(obj):
                    return True
        
        return False
    
    def is_empty(self):
        """检查四叉树是否为空"""
        if self.objects:
            return False
        
        if self.divided:
            for child in self.children:
                if not child.is_empty():
                    return False
        
        return True
    
    def clear(self):
        """清空四叉树"""
        self.objects.clear()
        
        if self.divided:
            for child in self.children:
                child.clear()
            
            self.children.clear()
            self.divided = False