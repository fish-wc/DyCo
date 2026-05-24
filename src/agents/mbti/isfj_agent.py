"""
ISFJ智能体 - 守护者
特点: 忠诚奉献,细心体贴,保护他人,注重传统
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.isfj.special_tool_example import hello_world

class ISFJAgent(BaseAgent):
    """ISFJ智能体 - 守护者"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
