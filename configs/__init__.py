"""
configs 包：DyCo 的配置加载入口。

用法：
    from configs import ConfigLoader, config_loader

    system_config = config_loader.load_system_config()   # 系统配置
    agent_ids     = config_loader.get_agent_ids()        # 默认组队
    agent_config  = config_loader.load_agent_config("entj_001")

ModelConfig 中的 `api_key` / `model_url` 填写的是环境变量名，
真实值请在 .env 或 shell 中设置（参见 .env.example）。
"""
from configs.loader import ConfigLoader, config_loader

__all__ = ["ConfigLoader", "config_loader"]
