"""
ENFP智能体 - 活动家
特点: 热情洋溢,富有创造力,善于激励他人,追求可能性
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.enfp.special_tool_example import hello_world

class ENFPAgent(BaseAgent):
    """ENFP智能体 - 活动家"""

    def __init__(self, config, message_manager, speaking_queue, logger = None, system_config = None, knowledge_manager = None):
        super().__init__(config, message_manager, speaking_queue, logger, system_config, knowledge_manager)