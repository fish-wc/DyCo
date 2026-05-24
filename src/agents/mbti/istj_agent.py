"""
ISTJ智能体 - 物流师
特点: 务实可靠,注重细节,遵守规则,高度负责
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.istj.special_tool_example import hello_world

class ISTJAgent(BaseAgent):
    """ISTJ智能体 - 物流师"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
