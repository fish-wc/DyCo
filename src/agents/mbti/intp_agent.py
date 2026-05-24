"""
INTP智能体 - 逻辑学家
特点: 理性分析,创新思维,喜欢理论,追求知识
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.intp.special_tool_example import hello_world

class INTPAgent(BaseAgent):
    """INTP智能体 - 逻辑学家"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
