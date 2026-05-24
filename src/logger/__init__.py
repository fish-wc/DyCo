"""
日志模块 - 简化版
提供全局logger实例和任务级别日志管理

使用示例:
    # 1. 直接使用全局logger
    from src.logger import logger
    logger.info("这是一条日志")
    
    # 2. 为任务创建专用logger
    from src.logger import setup_logger
    task_logger = setup_logger(task_id="task_123", workspace_root="./workspace")
    task_logger.info("任务日志")
    
"""
from .logger_config import (
    setup_logger,
)

__all__ = [
    'setup_logger',  # 为任务创建logger
]
