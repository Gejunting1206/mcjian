# 优化模块包

from .performance_optimizer import PerformanceOptimizer
from .comprehensive_performance_optimizer import ComprehensivePerformanceOptimizer
from .chunk_optimization import ChunkManager, InstancedRenderManager, MeshCombineManager, DistanceCheckOptimizer
from .frustum_culling import Frustum, FrustumCullingManager, frustum_culling_manager, get_visible_blocks
from .instanced_rendering import InstancedRenderer
from .advanced_shader_optimization import AdvancedShaderOptimizer
from .optimization_integration import OptimizationManager

__all__ = [
    'PerformanceOptimizer',
    'ComprehensivePerformanceOptimizer',
    'ChunkManager',
    'InstancedRenderManager',
    'MeshCombineManager',
    'DistanceCheckOptimizer',
    'Frustum',
    'FrustumCullingManager',
    'frustum_culling_manager',
    'get_visible_blocks',
    'InstancedRenderer',
    'AdvancedShaderOptimizer',
    'OptimizationManager'
]