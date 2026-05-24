"""
ENFJ智能体 - 主人公
特点: 魅力领袖,善于沟通,关注他人成长,富有感染力
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.enfj.special_tool_example import hello_world

class ENFJAgent(BaseAgent):
    """ENFJ智能体 - 主人公"""
    

    def __init__(self, config, message_manager, speaking_queue, logger = None, system_config = None, knowledge_manager = None):
        super().__init__(config, message_manager, speaking_queue, logger, system_config, knowledge_manager)