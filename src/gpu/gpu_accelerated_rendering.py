# GPU加速渲染模块
# 提供高效的GPU实例化渲染和着色器优化

from ursina import *
import numpy as np
import time
from collections import defaultdict

class GPUAcceleratedRenderer:
    """GPU加速渲染器 - 使用高级GPU技术提高渲染性能"""
    
    def __init__(self):
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒）
        
        # 实例化渲染参数
        self.instancing_enabled = True
        self.max_instances_per_batch = 1000  # 每批次最大实例数
        self.instance_groups = defaultdict(list)  # 按材质和模型分组
        self.instanced_entities = {}  # 存储实例化实体
        self.dirty_groups = set()  # 需要更新的组
        
        # 着色器优化参数
        self.shader_optimization_enabled = True
        self.custom_shaders = {}
        self.shader_cache = {}
        
        # 性能统计
        self.stats = {
            'draw_calls': 0,
            'instances_rendered': 0,
            'batches': 0,
            'render_time_ms': 0
        }
        
        # 初始化着色器
        self._init_shaders()
    
    def _init_shaders(self):
        """初始化优化着色器"""
        # 基础实例化着色器
        self.custom_shaders['instanced'] = Shader(
            vertex='''
            #version 150
            
            uniform mat4 p3d_ModelViewProjectionMatrix;
            uniform mat4 p3d_ModelMatrix;
            
            // 实例化数据
            in vec4 p3d_Vertex;
            in vec2 p3d_MultiTexCoord0;
            in vec3 p3d_Normal;
            in vec4 instanceColor;
            in vec3 instancePosition;
            in vec3 instanceScale;
            in vec4 instanceRotation;
            
            out vec2 texcoords;
            out vec3 world_normal;
            out vec4 vertex_color;
            
            // 四元数旋转函数
            vec3 quat_rotate(vec4 q, vec3 v) {
                vec3 qv = vec3(q.x, q.y, q.z);
                return v + 2.0 * cross(cross(v, qv) + q.w * v, qv);
            }
            
            void main() {
                // 应用实例变换
                vec3 transformed_vertex = p3d_Vertex.xyz * instanceScale;
                transformed_vertex = quat_rotate(instanceRotation, transformed_vertex);
                transformed_vertex += instancePosition;
                
                // 计算法线
                world_normal = quat_rotate(instanceRotation, p3d_Normal);
                
                // 传递纹理坐标和颜色
                texcoords = p3d_MultiTexCoord0;
                vertex_color = instanceColor;
                
                // 最终位置
                gl_Position = p3d_ModelViewProjectionMatrix * vec4(transformed_vertex, 1.0);
            }
            ''',
            fragment='''
            #version 150
            
            uniform sampler2D p3d_Texture0;
            uniform vec4 p3d_ColorScale;
            
            in vec2 texcoords;
            in vec3 world_normal;
            in vec4 vertex_color;
            
            out vec4 fragColor;
            
            void main() {
                vec4 tex_color = texture(p3d_Texture0, texcoords);
                
                // 简化的光照计算
                vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
                float ndotl = max(dot(normalize(world_normal), light_dir), 0.0);
                float lighting = 0.3 + ndotl * 0.7;  // 环境光 + 漫反射
                
                // 最终颜色
                fragColor = tex_color * vertex_color * vec4(lighting, lighting, lighting, 1.0) * p3d_ColorScale;
                
                // 丢弃透明像素
                if (fragColor.a < 0.1) discard;
            }
            '''
        )
        
        # 优化的地形着色器
        self.custom_shaders['terrain'] = Shader(
            vertex='''
            #version 150
            
            uniform mat4 p3d_ModelViewProjectionMatrix;
            uniform mat4 p3d_ModelMatrix;
            
            in vec4 p3d_Vertex;
            in vec2 p3d_MultiTexCoord0;
            in vec3 p3d_Normal;
            in vec4 p3d_Color;
            
            out vec2 texcoords;
            out vec3 world_normal;
            out vec3 world_position;
            out vec4 vertex_color;
            
            void main() {
                // 传递纹理坐标和颜色
                texcoords = p3d_MultiTexCoord0;
                vertex_color = p3d_Color;
                
                // 计算世界空间位置和法线
                world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
                world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
                
                // 最终位置
                gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            }
            ''',
            fragment='''
            #version 150
            
            uniform sampler2D p3d_Texture0;
            uniform vec4 p3d_ColorScale;
            uniform vec3 camera_position;
            uniform float fog_density;
            uniform vec3 fog_color;
            
            in vec2 texcoords;
            in vec3 world_normal;
            in vec3 world_position;
            in vec4 vertex_color;
            
            out vec4 fragColor;
            
            void main() {
                vec4 tex_color = texture(p3d_Texture0, texcoords);
                
                // 基础光照
                vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
                float ndotl = max(dot(normalize(world_normal), light_dir), 0.0);
                float lighting = 0.3 + ndotl * 0.7;  // 环境光 + 漫反射
                
                // 基础颜色
                vec4 base_color = tex_color * vertex_color * vec4(lighting, lighting, lighting, 1.0) * p3d_ColorScale;
                
                // 雾效
                float dist = distance(world_position, camera_position);
                float fog_factor = 1.0 - exp(-fog_density * dist);
                fog_factor = clamp(fog_factor, 0.0, 1.0);
                
                // 最终颜色（带雾效）
                fragColor = mix(base_color, vec4(fog_color, 1.0), fog_factor);
                
                // 丢弃透明像素
                if (fragColor.a < 0.1) discard;
            }
            '''
        )
    
    def add_entity(self, entity):
        """添加实体到GPU加速渲染系统"""
        if not self.enabled or not self.instancing_enabled:
            return
        
        # 创建实体的唯一标识（基于模型和材质）
        if hasattr(entity, 'model') and hasattr(entity, 'texture'):
            # 使用模型名称和纹理路径作为组标识
            model_name = entity.model.name if hasattr(entity.model, 'name') else str(entity.model)
            texture_path = entity.texture.path if hasattr(entity.texture, 'path') else str(entity.texture)
            group_id = f"{model_name}_{texture_path}"
            
            # 添加到实例组
            self.instance_groups[group_id].append(entity)
            self.dirty_groups.add(group_id)
    
    def remove_entity(self, entity):
        """从GPU加速渲染系统中移除实体"""
        if not self.enabled:
            return
        
        # 查找并移除实体
        for group_id, entities in self.instance_groups.items():
            if entity in entities:
                entities.remove(entity)
                self.dirty_groups.add(group_id)
                break
    
    def update(self, force=False):
        """更新GPU加速渲染"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # 降低更新频率
        if not force and current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        start_time = time.time()
        
        # 更新所有标记为脏的组
        dirty_groups_to_process = list(self.dirty_groups)[:5]  # 每次最多处理5个组
        
        batches = 0
        instances = 0
        
        for group_id in dirty_groups_to_process:
            entities = self.instance_groups[group_id]
            
            # 如果没有实体，销毁实例化实体并继续
            if not entities:
                if group_id in self.instanced_entities:
                    for instanced_entity in self.instanced_entities[group_id]:
                        destroy(instanced_entity)
                    del self.instanced_entities[group_id]
                self.dirty_groups.remove(group_id)
                continue
            
            # 销毁旧的实例化实体
            if group_id in self.instanced_entities:
                for instanced_entity in self.instanced_entities[group_id]:
                    destroy(instanced_entity)
            
            # 分批处理实体
            self.instanced_entities[group_id] = []
            
            # 按批次创建实例化实体
            for i in range(0, len(entities), self.max_instances_per_batch):
                batch = entities[i:i+self.max_instances_per_batch]
                batches += 1
                instances += len(batch)
                
                # 使用第一个实体的模型和纹理
                model = batch[0].model
                texture = batch[0].texture
                
                try:
                    # 创建实例化实体
                    instanced = Entity(model='cube')
                    
                    # 设置纹理和着色器
                    if texture:
                        instanced.texture = texture
                    
                    if self.shader_optimization_enabled:
                        instanced.shader = self.custom_shaders['instanced']
                    
                    # 合并实体，保留原始实体
                    instanced.combine(batch, auto_destroy=False, keep_origin=True)
                    
                    # 存储实例化实体
                    self.instanced_entities[group_id].append(instanced)
                    
                    # 隐藏原始实体的视觉效果，但保留碰撞体积
                    for entity in batch:
                        # 保存原始颜色和纹理
                        if not hasattr(entity, '_original_color'):
                            entity._original_color = entity.color
                            entity._original_texture = entity.texture
                        
                        # 隐藏视觉效果但保留碰撞
                        entity.color = color.clear
                        entity.texture = None
                except Exception as e:
                    print(f"GPU实例化渲染错误: {e}")
            
            # 从脏列表中移除
            self.dirty_groups.remove(group_id)
        
        # 更新统计信息
        self.stats['batches'] = batches
        self.stats['instances_rendered'] = instances
        self.stats['draw_calls'] = batches  # 简化估计
        self.stats['render_time_ms'] = (time.time() - start_time) * 1000
    
    def apply_shader_optimization(self, entity, shader_type='default'):
        """应用着色器优化到实体"""
        if not self.enabled or not self.shader_optimization_enabled:
            return
        
        if shader_type in self.custom_shaders:
            entity.shader = self.custom_shaders[shader_type]
    
    def restore_entities(self):
        """恢复所有实体的原始视觉效果"""
        for group_id, entities in self.instance_groups.items():
            for entity in entities:
                if hasattr(entity, '_original_color'):
                    entity.color = entity._original_color
                if hasattr(entity, '_original_texture'):
                    entity.texture = entity._original_texture
        
        # 销毁所有实例化实体
        for group_entities in self.instanced_entities.values():
            for entity in group_entities:
                destroy(entity)
        
        self.instanced_entities.clear()
        self.dirty_groups.clear()
    
    def toggle(self):
        """切换GPU加速渲染"""
        self.enabled = not self.enabled
        if not self.enabled:
            self.restore_entities()
        return self.enabled
    
    def toggle_instancing(self):
        """切换实例化渲染"""
        self.instancing_enabled = not self.instancing_enabled
        if not self.instancing_enabled:
            self.restore_entities()
        return self.instancing_enabled
    
    def toggle_shader_optimization(self):
        """切换着色器优化"""
        self.shader_optimization_enabled = not self.shader_optimization_enabled
        return self.shader_optimization_enabled

# 创建全局实例
gpu_renderer = GPUAcceleratedRenderer()