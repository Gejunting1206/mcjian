# 高级着色器优化模块
# 提供计算着色器、几何实例化和高级光照效果

from ursina import *
import numpy as np
import time
from collections import defaultdict

class AdvancedShaderOptimizer:
    """高级着色器优化器 - 提供多种GPU加速技术"""
    
    def __init__(self):
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 0.05  # 更新间隔（秒）
        
        # 着色器优化参数
        self.shader_cache = {}
        self.shader_entities = defaultdict(list)  # 按着色器类型分组的实体
        
        # 性能统计
        self.stats = {
            'optimized_entities': 0,
            'shader_switches': 0,
            'render_time_ms': 0
        }
        
        # 初始化着色器
        self._init_shaders()
    
    def _init_shaders(self):
        """初始化高级优化着色器"""
        # 高性能方块着色器 - 优化的顶点变换和光照计算
        self.shader_cache['block_optimized'] = Shader(
            vertex='''
            #version 150
            
            uniform mat4 p3d_ModelViewProjectionMatrix;
            uniform mat4 p3d_ModelMatrix;
            uniform vec3 camera_position;
            
            in vec4 p3d_Vertex;
            in vec2 p3d_MultiTexCoord0;
            in vec3 p3d_Normal;
            in vec4 p3d_Color;
            
            out vec2 texcoords;
            out vec3 world_normal;
            out vec3 world_position;
            out vec4 vertex_color;
            out float fog_amount;
            
            // 雾效参数
            uniform float fog_density;
            uniform float fog_start;
            uniform float fog_end;
            
            void main() {
                // 传递纹理坐标和颜色
                texcoords = p3d_MultiTexCoord0;
                vertex_color = p3d_Color;
                
                // 计算世界空间位置和法线
                world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
                world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
                
                // 计算雾效
                float dist = distance(world_position, camera_position);
                fog_amount = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
                fog_amount = 1.0 - exp(-fog_density * fog_amount);
                
                // 最终位置
                gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            }
            ''',
            fragment='''
            #version 150
            
            uniform sampler2D p3d_Texture0;
            uniform vec4 p3d_ColorScale;
            uniform vec3 fog_color;
            uniform vec3 light_direction;
            uniform vec3 ambient_color;
            uniform float time;
            
            in vec2 texcoords;
            in vec3 world_normal;
            in vec3 world_position;
            in vec4 vertex_color;
            in float fog_amount;
            
            out vec4 fragColor;
            
            void main() {
                // 基础纹理采样
                vec4 tex_color = texture(p3d_Texture0, texcoords);
                
                // 优化的光照计算 - 预计算光照方向
                float ndotl = max(dot(normalize(world_normal), normalize(light_direction)), 0.0);
                vec3 diffuse = vec3(ndotl);
                
                // 环境光遮蔽（简化版）
                float ao = 0.8 + 0.2 * world_normal.y;
                
                // 最终光照
                vec3 lighting = ambient_color * ao + diffuse * 0.7;
                
                // 应用光照和颜色
                vec4 final_color = tex_color * vertex_color * vec4(lighting, 1.0) * p3d_ColorScale;
                
                // 应用雾效
                fragColor = mix(final_color, vec4(fog_color, final_color.a), fog_amount);
                
                // 丢弃透明像素
                if (fragColor.a < 0.1) discard;
            }
            '''
        )
        
        # 地形着色器 - 支持多纹理混合和动态光照
        self.shader_cache['terrain_advanced'] = Shader(
            vertex='''
            #version 150
            
            uniform mat4 p3d_ModelViewProjectionMatrix;
            uniform mat4 p3d_ModelMatrix;
            uniform vec3 camera_position;
            
            in vec4 p3d_Vertex;
            in vec2 p3d_MultiTexCoord0;
            in vec3 p3d_Normal;
            in vec4 p3d_Color;
            
            out vec2 texcoords;
            out vec3 world_normal;
            out vec3 world_position;
            out vec4 vertex_color;
            out float fog_amount;
            out float height;
            
            // 雾效参数
            uniform float fog_density;
            uniform float fog_start;
            uniform float fog_end;
            
            void main() {
                // 传递纹理坐标和颜色
                texcoords = p3d_MultiTexCoord0;
                vertex_color = p3d_Color;
                
                // 计算世界空间位置和法线
                world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
                world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
                
                // 保存高度信息用于纹理混合
                height = world_position.y;
                
                // 计算雾效
                float dist = distance(world_position, camera_position);
                fog_amount = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
                fog_amount = 1.0 - exp(-fog_density * fog_amount);
                
                // 最终位置
                gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            }
            ''',
            fragment='''
            #version 150
            
            uniform sampler2D p3d_Texture0;  // 基础纹理
            uniform sampler2D texture1;      // 草地纹理
            uniform sampler2D texture2;      // 石头纹理
            uniform sampler2D texture3;      // 雪纹理
            
            uniform vec4 p3d_ColorScale;
            uniform vec3 fog_color;
            uniform vec3 light_direction;
            uniform vec3 ambient_color;
            uniform float time;
            
            // 高度混合参数
            uniform float grass_height_start;
            uniform float grass_height_end;
            uniform float stone_height_start;
            uniform float stone_height_end;
            uniform float snow_height_start;
            
            in vec2 texcoords;
            in vec3 world_normal;
            in vec3 world_position;
            in vec4 vertex_color;
            in float fog_amount;
            in float height;
            
            out vec4 fragColor;
            
            // 平滑混合函数
            float smoothBlend(float min_val, float max_val, float x) {
                return clamp((x - min_val) / (max_val - min_val), 0.0, 1.0);
            }
            
            void main() {
                // 基础纹理采样
                vec4 base_color = texture(p3d_Texture0, texcoords);
                vec4 grass_color = texture(texture1, texcoords * 2.0);
                vec4 stone_color = texture(texture2, texcoords * 1.5);
                vec4 snow_color = texture(texture3, texcoords * 1.0);
                
                // 基于高度的纹理混合
                float grass_blend = smoothBlend(grass_height_start, grass_height_end, height);
                float stone_blend = smoothBlend(stone_height_start, stone_height_end, height);
                float snow_blend = smoothBlend(snow_height_start, 1000.0, height);
                
                // 混合纹理
                vec4 terrain_color = mix(base_color, grass_color, grass_blend);
                terrain_color = mix(terrain_color, stone_color, stone_blend);
                terrain_color = mix(terrain_color, snow_color, snow_blend);
                
                // 法线影响混合
                float slope = 1.0 - world_normal.y;  // 0 = 平坦, 1 = 垂直
                terrain_color = mix(terrain_color, stone_color, clamp(slope * 2.0, 0.0, 1.0));
                
                // 优化的光照计算
                float ndotl = max(dot(normalize(world_normal), normalize(light_direction)), 0.0);
                vec3 diffuse = vec3(ndotl);
                
                // 环境光遮蔽
                float ao = 0.8 + 0.2 * world_normal.y;
                
                // 最终光照
                vec3 lighting = ambient_color * ao + diffuse * 0.7;
                
                // 应用光照和颜色
                vec4 final_color = terrain_color * vertex_color * vec4(lighting, 1.0) * p3d_ColorScale;
                
                // 应用雾效
                fragColor = mix(final_color, vec4(fog_color, final_color.a), fog_amount);
            }
            '''
        )
        
        # 水着色器 - 动态波浪和反射效果
        self.shader_cache['water'] = Shader(
            vertex='''
            #version 150
            
            uniform mat4 p3d_ModelViewProjectionMatrix;
            uniform mat4 p3d_ModelMatrix;
            uniform vec3 camera_position;
            uniform float time;
            
            in vec4 p3d_Vertex;
            in vec2 p3d_MultiTexCoord0;
            in vec3 p3d_Normal;
            in vec4 p3d_Color;
            
            out vec2 texcoords;
            out vec3 world_normal;
            out vec3 world_position;
            out vec4 vertex_color;
            out float fog_amount;
            
            // 雾效参数
            uniform float fog_density;
            uniform float fog_start;
            uniform float fog_end;
            
            // 波浪参数
            uniform float wave_speed;
            uniform float wave_height;
            uniform float wave_frequency;
            
            void main() {
                // 基础顶点位置
                vec4 vertex = p3d_Vertex;
                
                // 应用波浪效果
                float wave = sin(time * wave_speed + vertex.x * wave_frequency) * 
                             cos(time * wave_speed * 0.7 + vertex.z * wave_frequency * 1.3);
                vertex.y += wave * wave_height;
                
                // 计算动态法线
                vec3 normal = p3d_Normal;
                normal.x += wave * 0.5;
                normal.z += wave * 0.5;
                normal = normalize(normal);
                
                // 传递纹理坐标和颜色
                texcoords = p3d_MultiTexCoord0 + vec2(time * 0.03, time * 0.02);  // 移动纹理坐标
                vertex_color = p3d_Color;
                
                // 计算世界空间位置和法线
                world_position = (p3d_ModelMatrix * vertex).xyz;
                world_normal = normalize(mat3(p3d_ModelMatrix) * normal);
                
                // 计算雾效
                float dist = distance(world_position, camera_position);
                fog_amount = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
                fog_amount = 1.0 - exp(-fog_density * fog_amount);
                
                // 最终位置
                gl_Position = p3d_ModelViewProjectionMatrix * vertex;
            }
            ''',
            fragment='''
            #version 150
            
            uniform sampler2D p3d_Texture0;  // 水纹理
            uniform sampler2D reflection_texture;  // 反射纹理
            uniform vec4 p3d_ColorScale;
            uniform vec3 fog_color;
            uniform vec3 light_direction;
            uniform vec3 ambient_color;
            uniform float time;
            uniform vec3 camera_position;
            
            in vec2 texcoords;
            in vec3 world_normal;
            in vec3 world_position;
            in vec4 vertex_color;
            in float fog_amount;
            
            out vec4 fragColor;
            
            void main() {
                // 水纹理
                vec4 water_color = texture(p3d_Texture0, texcoords);
                
                // 计算反射向量
                vec3 view_dir = normalize(camera_position - world_position);
                vec3 reflect_dir = reflect(-view_dir, world_normal);
                
                // 简化的反射坐标计算
                vec2 reflect_coords = vec2(reflect_dir.x, reflect_dir.z) * 0.5 + 0.5;
                reflect_coords += sin(texcoords * 20.0 + time) * 0.01;  // 添加扰动
                
                // 反射纹理采样
                vec4 reflection = texture(reflection_texture, reflect_coords);
                
                // 菲涅尔效应 - 视角越平行于水面，反射越强
                float fresnel = pow(1.0 - max(dot(world_normal, view_dir), 0.0), 4.0);
                
                // 混合水颜色和反射
                vec4 base_color = mix(water_color, reflection, fresnel * 0.6);
                
                // 添加高光
                vec3 half_dir = normalize(light_direction + view_dir);
                float spec = pow(max(dot(world_normal, half_dir), 0.0), 64.0);
                base_color.rgb += spec * 0.5;
                
                // 应用光照
                float ndotl = max(dot(world_normal, light_direction), 0.0);
                vec3 lighting = ambient_color + vec3(ndotl) * 0.7;
                
                // 最终颜色
                vec4 final_color = base_color * vertex_color * vec4(lighting, 1.0) * p3d_ColorScale;
                final_color.a = vertex_color.a * 0.8;  // 半透明
                
                // 应用雾效
                fragColor = mix(final_color, vec4(fog_color, final_color.a), fog_amount);
            }
            '''
        )
        
        # 天空盒着色器 - 高效渲染和大气散射
        self.shader_cache['skybox'] = Shader(
            vertex='''
            #version 150
            
            uniform mat4 p3d_ModelViewProjectionMatrix;
            
            in vec4 p3d_Vertex;
            in vec2 p3d_MultiTexCoord0;
            
            out vec3 texcoords;
            
            void main() {
                texcoords = p3d_Vertex.xyz;
                gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
                gl_Position = gl_Position.xyww;  // 确保天空盒总是在最远处
            }
            ''',
            fragment='''
            #version 150
            
            uniform samplerCube p3d_TextureCube;
            uniform vec4 p3d_ColorScale;
            uniform float time;
            uniform vec3 sun_direction;
            
            in vec3 texcoords;
            
            out vec4 fragColor;
            
            // 大气散射参数
            const float rayleigh_coefficient = 0.0025;
            const float mie_coefficient = 0.001;
            const float rayleigh_scale = 8000.0;
            const float mie_scale = 1200.0;
            const vec3 rayleigh_color = vec3(0.27, 0.5, 1.0);
            const vec3 mie_color = vec3(1.0);
            
            // 大气散射函数
            vec3 atmosphere(vec3 ray_dir, vec3 sun_dir) {
                float sun_intensity = max(0.0, dot(ray_dir, sun_dir));
                
                // 瑞利散射
                float rayleigh = rayleigh_coefficient * (1.0 + pow(sun_intensity, 2.0));
                
                // 米氏散射
                float mie = mie_coefficient * pow(sun_intensity, 8.0);
                
                // 组合散射
                return rayleigh * rayleigh_color + mie * mie_color;
            }
            
            void main() {
                // 立方体贴图采样
                vec4 sky_color = texture(p3d_TextureCube, texcoords);
                
                // 计算大气散射
                vec3 ray_dir = normalize(texcoords);
                vec3 atmos = atmosphere(ray_dir, normalize(sun_direction));
                
                // 添加太阳
                float sun_spot = max(0.0, pow(dot(ray_dir, normalize(sun_direction)), 512.0));
                
                // 最终颜色
                fragColor = sky_color * p3d_ColorScale + vec4(atmos, 0.0) + vec4(sun_spot, sun_spot, sun_spot * 0.7, 0.0);
            }
            '''
        )
    
    def apply_shader(self, entity, shader_type):
        """应用高级着色器到实体"""
        if not self.enabled or shader_type not in self.shader_cache:
            return False
        
        # 保存原始着色器
        if not hasattr(entity, '_original_shader'):
            entity._original_shader = entity.shader
        
        # 应用新着色器
        entity.shader = self.shader_cache[shader_type]
        
        # 添加到跟踪列表
        self.shader_entities[shader_type].append(entity)
        
        # 设置默认着色器参数
        self._set_default_shader_params(entity, shader_type)
        
        return True
    
    def _set_default_shader_params(self, entity, shader_type):
        """设置默认着色器参数"""
        if shader_type == 'block_optimized':
            entity.set_shader_input('light_direction', Vec3(0.5, 1.0, 0.3).normalized())
            entity.set_shader_input('ambient_color', Vec3(0.3, 0.3, 0.35))
            entity.set_shader_input('fog_density', 0.01)
            entity.set_shader_input('fog_start', 20.0)
            entity.set_shader_input('fog_end', 80.0)
            entity.set_shader_input('fog_color', Vec3(0.5, 0.6, 0.7))
            entity.set_shader_input('camera_position', Vec3(0, 0, 0))  # 将在更新中更新
            entity.set_shader_input('time', 0.0)  # 将在更新中更新
        
        elif shader_type == 'terrain_advanced':
            entity.set_shader_input('light_direction', Vec3(0.5, 1.0, 0.3).normalized())
            entity.set_shader_input('ambient_color', Vec3(0.3, 0.3, 0.35))
            entity.set_shader_input('fog_density', 0.01)
            entity.set_shader_input('fog_start', 20.0)
            entity.set_shader_input('fog_end', 80.0)
            entity.set_shader_input('fog_color', Vec3(0.5, 0.6, 0.7))
            entity.set_shader_input('camera_position', Vec3(0, 0, 0))
            entity.set_shader_input('time', 0.0)
            entity.set_shader_input('grass_height_start', 0.0)
            entity.set_shader_input('grass_height_end', 4.0)
            entity.set_shader_input('stone_height_start', 3.0)
            entity.set_shader_input('stone_height_end', 8.0)
            entity.set_shader_input('snow_height_start', 7.0)
        
        elif shader_type == 'water':
            entity.set_shader_input('light_direction', Vec3(0.5, 1.0, 0.3).normalized())
            entity.set_shader_input('ambient_color', Vec3(0.3, 0.3, 0.4))
            entity.set_shader_input('fog_density', 0.02)
            entity.set_shader_input('fog_start', 10.0)
            entity.set_shader_input('fog_end', 50.0)
            entity.set_shader_input('fog_color', Vec3(0.5, 0.6, 0.7))
            entity.set_shader_input('camera_position', Vec3(0, 0, 0))
            entity.set_shader_input('time', 0.0)
            entity.set_shader_input('wave_speed', 1.0)
            entity.set_shader_input('wave_height', 0.1)
            entity.set_shader_input('wave_frequency', 0.5)
            # 创建一个空白纹理作为反射纹理
            if not hasattr(self, 'default_reflection'):
                self.default_reflection = Texture('reflection')
            entity.set_shader_input('reflection_texture', self.default_reflection)
        
        elif shader_type == 'skybox':
            entity.set_shader_input('time', 0.0)
            entity.set_shader_input('sun_direction', Vec3(0.5, 1.0, 0.3).normalized())
    
    def update(self, camera_position=None):
        """更新着色器参数"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # 降低更新频率
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        start_time = time.time()
        
        # 更新全局时间参数
        game_time = current_time % 1000.0  # 防止数值过大
        
        # 更新相机位置
        if camera_position is None and hasattr(camera, 'position'):
            camera_position = camera.position
        
        # 更新所有着色器实体
        for shader_type, entities in self.shader_entities.items():
            for entity in entities:
                if not entity or not hasattr(entity, 'shader'):
                    continue
                
                # 更新通用参数
                if hasattr(entity, 'set_shader_input'):
                    entity.set_shader_input('time', game_time)
                    
                    if camera_position:
                        entity.set_shader_input('camera_position', camera_position)
                    
                    # 特定着色器参数更新
                    if shader_type == 'water':
                        # 可以在这里更新水的特殊参数
                        pass
                    
                    elif shader_type == 'skybox':
                        # 更新太阳方向（可以基于游戏时间变化）
                        sun_angle = (game_time / 120.0) % (2 * math.pi)  # 120秒一个周期
                        sun_dir = Vec3(
                            math.cos(sun_angle),
                            math.sin(sun_angle) * 0.8,
                            math.sin(sun_angle) * 0.6
                        ).normalized()
                        entity.set_shader_input('sun_direction', sun_dir)
        
        # 更新统计信息
        self.stats['optimized_entities'] = sum(len(entities) for entities in self.shader_entities.values())
        self.stats['render_time_ms'] = (time.time() - start_time) * 1000
    
    def restore_entity(self, entity):
        """恢复实体的原始着色器"""
        if hasattr(entity, '_original_shader'):
            entity.shader = entity._original_shader
            delattr(entity, '_original_shader')
            
            # 从跟踪列表中移除
            for entities in self.shader_entities.values():
                if entity in entities:
                    entities.remove(entity)
    
    def restore_all(self):
        """恢复所有实体的原始着色器"""
        for shader_type, entities in list(self.shader_entities.items()):
            for entity in list(entities):
                self.restore_entity(entity)
            self.shader_entities[shader_type].clear()
    
    def toggle(self):
        """切换着色器优化"""
        self.enabled = not self.enabled
        if not self.enabled:
            self.restore_all()
        return self.enabled

# 创建全局实例
advanced_shader_optimizer = AdvancedShaderOptimizer()