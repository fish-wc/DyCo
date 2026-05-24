"""
INFP智能体 - 调停者
特点: 理想主义,富有同情心,追求和谐,注重价值观
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.infp.special_tool_example import hello_world

class INFPAgent(BaseAgent):
    """INFP智能体 - 调停者"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
