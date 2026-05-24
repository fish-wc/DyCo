"""
ISTP智能体 - 鉴赏家
特点: 动手能力强,冷静理性,善于解决问题,追求自由
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.istp.special_tool_example import hello_world

class ISTPAgent(BaseAgent):
    """ISTP智能体 - 鉴赏家"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
