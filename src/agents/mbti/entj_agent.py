"""
ENTJ智能体 - 指挥官
特点: 领导力强,果断高效,目标导向,善于组织
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.entj.special_tool_example import hello_world

class ENTJAgent(BaseAgent):
    """ENTJ智能体 - 指挥官"""
    

    def __init__(self, config, message_manager, speaking_queue, logger = None, system_config = None, knowledge_manager = None):
        super().__init__(config, message_manager, speaking_queue, logger, system_config, knowledge_manager)