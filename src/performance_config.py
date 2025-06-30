# 性能优化配置文件
# 集中管理所有性能相关的参数和设置

class PerformanceConfig:
    """性能优化配置类 - 集中管理所有性能参数"""
    
    # 目标帧率设置
    TARGET_FPS = 60
    MIN_ACCEPTABLE_FPS = 30
    
    # 更新频率控制（帧数间隔）
    PERFORMANCE_OPTIMIZER_UPDATE_INTERVAL = 8  # 每8帧更新一次性能优化器（从12减少到8）
    FRUSTUM_CULLING_UPDATE_INTERVAL = 8        # 每8帧更新一次视锥剔除（从16减少到8）
    MESH_RENDER_INTERVAL = 2                    # 每2帧渲染一次（从4减少到2）
    BACKUP_RENDER_INTERVAL = 32                 # 每32帧执行一次备份渲染（从64减少到32）
    
    # 物理检测频率
    PLAYER_STUCK_CHECK_INTERVAL = 12            # 每12帧检查玩家是否卡住（从24减少到12）
    PLAYER_LANDING_CHECK_INTERVAL = 8          # 每8帧检查玩家落地（从16减少到8）
    
    # 区块管理频率
    CHUNK_UPDATE_INTERVAL = 32                  # 每32帧更新区块（从64减少到32）
    CHUNK_UNLOAD_CHECK_INTERVAL = 15            # 每15帧检查区块卸载（从30减少到15）
    
    # UI更新频率
    FPS_DISPLAY_UPDATE_INTERVAL = 8             # 每8次FPS计算更新一次显示
    PERFORMANCE_STATS_UPDATE_INTERVAL = 8       # 每8次FPS计算更新一次性能统计
    OPTIMIZATION_MANAGER_UPDATE_INTERVAL = 16   # 每16次FPS计算更新一次优化管理器
    
    # 渲染优化参数
    CAMERA_MOVE_THRESHOLD = 3.0                 # 摄像机移动阈值（从5.0减少到3.0）
    RENDER_DISTANCE_LIMIT = 16                  # 渲染距离限制（从40减少到16）
    SPATIAL_HASH_CELL_SIZE = 64                 # 空间哈希网格大小
    
    # 批量渲染参数
    MAX_BATCH_SIZE = 100                        # 最大批量渲染数量
    BATCH_RENDER_INTERVAL = 2                   # 批量渲染间隔（从4减少到2）
    
    # 异步任务参数
    MAX_ASYNC_TASKS = 10                        # 最大异步任务数量
    TASK_TIMEOUT = 5.0                          # 任务超时时间（秒）
    
    # LOD（细节层次）参数
    LOD_DISTANCE_NEAR = 20                      # 近距离LOD阈值
    LOD_DISTANCE_FAR = 50                       # 远距离LOD阈值
    
    # 内存管理参数
    MAX_LOADED_CHUNKS = 100                     # 最大加载区块数量
    CHUNK_CACHE_SIZE = 50                       # 区块缓存大小
    
    # 输入处理参数
    INPUT_PROCESSING_INTERVAL = 1               # 输入处理间隔（新增参数）
    MOUSE_SMOOTHING_FACTOR = 0.8                # 鼠标平滑因子（新增参数）
    KEYBOARD_RESPONSE_DELAY = 0.01              # 键盘响应延迟（秒）（新增参数）
    
    @classmethod
    def get_adaptive_settings(cls, current_fps):
        """根据当前帧率自适应调整设置"""
        if current_fps < 20:
            # 极低帧率：最激进的优化
            return {
                'render_interval': 4,            # 从8减少到4
                'update_interval': 16,           # 从24减少到16
                'chunk_interval': 64,            # 从128减少到64
                'physics_interval': 24           # 从48减少到24
            }
        elif current_fps < 30:
            # 低帧率：激进优化
            return {
                'render_interval': 3,            # 从6减少到3
                'update_interval': 12,           # 从18减少到12
                'chunk_interval': 48,            # 从96减少到48
                'physics_interval': 16           # 从32减少到16
            }
        elif current_fps < 45:
            # 中等帧率：平衡优化
            return {
                'render_interval': 2,            # 从4减少到2
                'update_interval': 8,            # 从12减少到8
                'chunk_interval': 32,            # 从64减少到32
                'physics_interval': 12           # 从24减少到12
            }
        else:
            # 高帧率：轻度优化
            return {
                'render_interval': 1,            # 从2减少到1
                'update_interval': 4,            # 从8减少到4
                'chunk_interval': 16,            # 从32减少到16
                'physics_interval': 8            # 从16减少到8
            }
    
    @classmethod
    def apply_extreme_optimization(cls):
        """应用极限优化设置"""
        cls.PERFORMANCE_OPTIMIZER_UPDATE_INTERVAL = 12  # 从20减少到12
        cls.FRUSTUM_CULLING_UPDATE_INTERVAL = 16       # 从32减少到16
        cls.MESH_RENDER_INTERVAL = 4                   # 从8减少到4
        cls.BACKUP_RENDER_INTERVAL = 64                # 从128减少到64
        cls.PLAYER_STUCK_CHECK_INTERVAL = 24           # 从48减少到24
        cls.PLAYER_LANDING_CHECK_INTERVAL = 16         # 从32减少到16
        cls.CHUNK_UPDATE_INTERVAL = 64                 # 从128减少到64
        cls.FPS_DISPLAY_UPDATE_INTERVAL = 8            # 从16减少到8
        cls.OPTIMIZATION_MANAGER_UPDATE_INTERVAL = 16  # 从32减少到16
        
        print("已应用极限优化设置 - 最大化帧率")
        
    @classmethod
    def apply_input_optimization(cls):
        """应用输入优化设置，提高操作流畅度"""
        cls.INPUT_PROCESSING_INTERVAL = 1              # 确保每帧处理输入
        cls.MOUSE_SMOOTHING_FACTOR = 0.7              # 降低平滑因子，提高响应速度
        cls.KEYBOARD_RESPONSE_DELAY = 0.005           # 减少键盘响应延迟
        
        print("已应用输入优化设置 - 提高操作流畅度")

# 全局性能配置实例
perf_config = PerformanceConfig()

# 自动应用极限优化
perf_config.apply_extreme_optimization()
perf_config.apply_input_optimization()