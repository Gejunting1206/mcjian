"""
备份查询模块
根据输入描述生成所有代码的备份
"""
import os
import shutil
import fnmatch
from datetime import datetime
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='backup.log',
    encoding='utf-8'
)

class BackupQuery:
    """备份查询类，用于根据描述生成代码备份"""
    
    def __init__(self, code_dir, backup_dir, description):
        """
        初始化备份查询
        :param code_dir: 代码目录路径
        :param backup_dir: 备份目录路径
        :param description: 版本描述
        """
        self.code_dir = code_dir
        self.backup_dir = backup_dir + '\\' + description
        
        # 确保备份目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def query_and_backup(self, description_pattern):
        """
        根据描述模式查询代码文件并生成备份
        :param description_pattern: 描述模式(支持通配符)
        :return: 备份文件路径列表
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = os.path.join(self.backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_folder, exist_ok=True)
        
        # 遍历根目录查找匹配文件
        matched_files = []
        for file in os.listdir(self.code_dir):
            if fnmatch.fnmatch(file, "*.py"):
                src_path = os.path.join(self.code_dir, file)
                dst_path = os.path.join(backup_folder, file)
                try:
                    # 尝试打开文件检测是否被锁定
                    with open(src_path, 'rb') as f:
                        pass
                    shutil.copy2(src_path, dst_path)
                    matched_files.append(dst_path)
                    logging.info(f"成功备份文件: {src_path} -> {dst_path}")
                except (PermissionError, IOError) as e:
                    logging.error(f"无法备份文件 {src_path}: {e}")
                    continue
        
        return matched_files

# 示例用法
if __name__ == "__main__":
    # 配置代码目录和备份目录
    s = input("描述:")

    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKUP_DIR = os.path.join(CODE_DIR, "backups")
    
    # 创建备份查询实例
    backup_query = BackupQuery('.//', BACKUP_DIR, s)
    
    # 根据描述生成备份
    description = "*.py"
    backed_up_files = backup_query.query_and_backup(description)
    
    # 输出结果
    print(f"已生成 {len(backed_up_files)} 个备份文件:")
    for file in backed_up_files:
        print(f"- {file}")