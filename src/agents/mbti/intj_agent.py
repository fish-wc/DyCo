"""
INTJ智能体 - 架构师
特点: 战略思维,独立性强,善于规划,追求完美
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.intj.special_tool_example import hello_world

class INTJAgent(BaseAgent):
    """INTJ智能体 - 架构师"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
