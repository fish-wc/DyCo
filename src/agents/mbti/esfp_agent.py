"""
ESFP智能体 - 表演者
特点: 活力四射,善于娱乐,热爱生活,富有表现力
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.esfp.special_tool_example import hello_world

class ESFPAgent(BaseAgent):
    """ESFP智能体 - 表演者"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
