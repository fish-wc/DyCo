"""
项目日志系统 - 简化版
提供全局日志记录器和任务级别的日志管理
"""
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


class TaskRotatingFileHandler(RotatingFileHandler):
    """
    自动轮转文件处理器
    当日志文件超过指定大小时，自动创建新文件
    """
    
    def __init__(self, filename, mode='a', maxBytes=5*1024*1024, 
                 backupCount=10, encoding='utf-8', delay=False):
        """
        初始化处理器
        
        Args:
            filename: 日志文件路径
            mode: 文件打开模式
            maxBytes: 单个日志文件最大字节数 (默认5MB)
            backupCount: 保留的旧日志文件数量
            encoding: 文件编码
            delay: 是否延迟创建文件
        """
        # 确保目录存在
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            filename=filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )
    
    def doRollover(self):
        """
        执行日志轮转
        使用时间戳命名备份文件
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # 使用时间戳命名备份文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(self.baseFilename)
        backup_name = f"{base_path.stem}_{timestamp}{base_path.suffix}"
        backup_path = base_path.parent / backup_name
        
        # 重命名当前文件
        if base_path.exists():
            base_path.rename(backup_path)
        
        # 清理旧的备份文件
        self._cleanup_old_backups(base_path)
        
        # 打开新文件
        if not self.delay:
            self.stream = self._open()
    
    def _cleanup_old_backups(self, base_path: Path):
        """清理超过保留数量的旧备份文件"""
        pattern = f"{base_path.stem}_*{base_path.suffix}"
        backup_files = sorted(
            base_path.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # 删除超过backupCount的旧文件
        for old_file in backup_files[self.backupCount:]:
            try:
                old_file.unlink()
            except Exception:
                pass

def setup_logger(task_id: str, workspace_root: str,model_name:str=None, 
                     log_level=logging.DEBUG, max_bytes=5*1024*1024, backup_count=10):
    """
    为任务创建日志记录器
    日志文件保存在 workspace/messages/<task_id>/task.log（与messages.json同级）
    
    Args:
        task_id: 任务ID
        workspace_root: 工作空间根目录
        log_level: 日志级别
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的旧日志文件数量
        
    Returns:
        logger实例
    """
    if model_name is None: 
        # 构建日志文件路径（与messages.json同级）
        log_dir = Path(workspace_root) / "messages" / task_id

    else:
        # 构建日志文件路径（与messages.json同级）
        log_dir = Path(workspace_root) / "messages" / model_name / task_id

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "task.log"
    # 创建任务专用logger
    logger_name = f"task.{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # 不传播到父logger
    
    # 清除已有handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件handler（支持自动分割）
    file_handler = TaskRotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"任务日志记录器已初始化: {task_id}")
    logger.info(f"日志文件: {log_file}")
    
    return logger


