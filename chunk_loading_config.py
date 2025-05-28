# 区块加载和缓存配置
# 提供可调整的参数以优化性能

class ChunkLoadingConfig:
    """区块加载和缓存配置类"""
    
    # 区块加载参数
    MAX_CHUNKS_PER_FRAME = 8        # 每帧最大加载区块数 (从16降低到8)
    PRELOAD_DISTANCE = 1            # 预加载距离 (从2降低到1)
    UNLOAD_DISTANCE = 3             # 卸载距离 (从4降低到3)
    LOADING_THREADS = 2             # 加载线程数 (从4降低到2)
    
    # 缓存参数
    CACHE_SIZE = 100                # 缓存大小 (从150降低到100)
    MESH_CACHE_WEIGHT = 0.6         # 网格缓存权重
    DATA_CACHE_WEIGHT = 0.3         # 数据缓存权重
    COLLISION_CACHE_WEIGHT = 0.1    # 碰撞数据缓存权重
    
    # 性能优化参数
    UPDATE_INTERVAL = 0.02          # 更新间隔(秒) (从0.01增加到0.02)
    ADAPTIVE_LOADING = True         # 启用自适应加载
    MIN_FPS_THRESHOLD = 30          # 最低FPS阈值
    
    # 预加载优化
    PRIORITIZE_VISIBLE = True       # 优先加载可见区块
    PRIORITIZE_PLAYER_PATH = True   # 优先加载玩家路径上的区块
    
    # 内存管理
    MAX_MEMORY_USAGE = 384 * 1024 * 1024  # 最大内存使用量(从512MB降低到384MB)
    MEMORY_CLEANUP_THRESHOLD = 0.8   # 内存清理阈值(从90%降低到80%)
    
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