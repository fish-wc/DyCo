"""
ESTP智能体 - 企业家
特点: 行动力强,敢于冒险,善于应变,享受挑战
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.estp.special_tool_example import hello_world

class ESTPAgent(BaseAgent):
    """ESTP智能体 - 企业家"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
