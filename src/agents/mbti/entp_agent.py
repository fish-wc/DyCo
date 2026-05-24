"""
ENTP智能体 - 辩论家
特点: 思维敏捷,喜欢辩论,创新精神,挑战权威
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.entp.special_tool_example import hello_world

class ENTPAgent(BaseAgent):
    """ENTP智能体 - 辩论家"""

    def __init__(self, config, message_manager, speaking_queue, logger = None, system_config = None, knowledge_manager = None):
        super().__init__(config, message_manager, speaking_queue, logger, system_config, knowledge_manager)