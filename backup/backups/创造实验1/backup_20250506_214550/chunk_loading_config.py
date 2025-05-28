# 区块加载和缓存配置
# 提供可调整的参数以优化性能

class ChunkLoadingConfig:
    """区块加载和缓存配置类"""
    
    # 区块加载参数
    MAX_CHUNKS_PER_FRAME = 16       # 每帧最大加载区块数 (大幅提高)
    PRELOAD_DISTANCE = 2            # 预加载距离
    UNLOAD_DISTANCE = 4             # 卸载距离
    LOADING_THREADS = 4             # 加载线程数 (增加)
    
    # 缓存参数
    CACHE_SIZE = 150                # 缓存大小
    MESH_CACHE_WEIGHT = 0.6         # 网格缓存权重
    DATA_CACHE_WEIGHT = 0.3         # 数据缓存权重
    COLLISION_CACHE_WEIGHT = 0.1    # 碰撞数据缓存权重
    
    # 性能优化参数
    UPDATE_INTERVAL = 0.01          # 更新间隔(秒) (显著缩短)
    ADAPTIVE_LOADING = True         # 启用自适应加载
    MIN_FPS_THRESHOLD = 30          # 最低FPS阈值
    
    # 预加载优化
    PRIORITIZE_VISIBLE = True       # 优先加载可见区块
    PRIORITIZE_PLAYER_PATH = True   # 优先加载玩家路径上的区块
    
    # 内存管理
    MAX_MEMORY_USAGE = 512 * 1024 * 1024  # 最大内存使用量(512MB)
    MEMORY_CLEANUP_THRESHOLD = 0.9   # 内存清理阈值(90%)
    
    # 缓存淘汰策略权重
    RECENCY_WEIGHT = 0.7            # 最近使用权重
    FREQUENCY_WEIGHT = 0.3          # 使用频率权重
    
    @classmethod
    def get_cache_distribution(cls):
        """获取各类型缓存的大小分配"""
        return {
            'mesh': int(cls.CACHE_SIZE * cls.MESH_CACHE_WEIGHT),
            'data': int(cls.CACHE_SIZE * cls.DATA_CACHE_WEIGHT),
            'collision': int(cls.CACHE_SIZE * cls.COLLISION_CACHE_WEIGHT)
        }
    
    @classmethod
    def should_cleanup_memory(cls, current_usage):
        """判断是否需要进行内存清理"""
        return current_usage > (cls.MAX_MEMORY_USAGE * cls.MEMORY_CLEANUP_THRESHOLD)