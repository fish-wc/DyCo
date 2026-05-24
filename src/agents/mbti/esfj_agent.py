"""
ESFJ智能体 - 执政官
特点: 热心助人,善于社交,注重和谐,责任心强
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.esfj.special_tool_example import hello_world

class ESFJAgent(BaseAgent):
    """ESFJ智能体 - 执政官"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
