from src.logger.logger_config import setup_logger
from pathlib import Path
import logging

task_id="common"
workspace_root="workspace"

# 实例化一个通用日志记录器
common_logger = setup_logger(
    task_id=task_id,
    workspace_root=workspace_root
)

# 添加控制台处理器（使日志同时输出到控制台和文件）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上级别
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
common_logger.addHandler(console_handler)

def clear_log():
    log_dir = Path(workspace_root) / "messages" / task_id
    # 清理该目录下所有的.log 文件
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            try:
                log_file.unlink()
                print(f"Removed log file: {log_file}")
            except Exception as e:
                print(f"Failed to remove log file {log_file}: {e}")
    else:
        print(f"Log directory does not exist: {log_dir}")
