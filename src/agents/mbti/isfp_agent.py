"""
ISFP智能体 - 冒险家
特点: 温和友善,热爱艺术,活在当下,追求美感
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.isfp.special_tool_example import hello_world

class ISFPAgent(BaseAgent):
    """ISFP智能体 - 冒险家"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
