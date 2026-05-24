"""
ESTJ智能体 - 管理者
特点: 组织能力强,实际高效,坚持原则,善于管理
"""
from ..base_agent import BaseAgent

# 这里导入各种工具
from ...tools.mbti.estj.special_tool_example import hello_world

class ESTJAgent(BaseAgent):
    """ESTJ智能体 - 管理者"""
    
    def register_tools(self):
        tools = [
            hello_world,  # 一个添加工具的简单测试
        ]
        
        self._register_smolagent_tool(tools)
