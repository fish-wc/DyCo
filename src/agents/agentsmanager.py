"""
用来管理agent的，这里设定了默认agent加载参数，便于直接进行调用。
"""
import os
from typing import Dict, Type

from configs import config_loader
from src.agents import (BaseAgent,INTJAgent,INTPAgent,ENTJAgent,ENTPAgent,INFJAgent,
                        INFPAgent,ENFJAgent,ENFPAgent,ISTJAgent,ISFJAgent,ESTJAgent,
                        ESFJAgent,ISTPAgent,ISFPAgent,ESTPAgent,ESFPAgent)
from src.communication.message_manager import MessageManager
from src.communication.speaking_queue import SpeakingQueue
from src.logger.logger_config import setup_logger
from src.tools.system.knowledgemanager import init_knowledge_manager
from src.utils.llm_client import create_llm_client

AGENT_CLASS_MAP: Dict[str, Type[BaseAgent]]= {
            'intj': INTJAgent,
            'intp': INTPAgent,
            'entj': ENTJAgent,
            'entp': ENTPAgent,
            'infj': INFJAgent,
            'infp': INFPAgent,
            'enfj': ENFJAgent,
            'enfp': ENFPAgent,
            'istj': ISTJAgent,
            'isfj': ISFJAgent,
            'estj': ESTJAgent,
            'esfj': ESFJAgent,
            'istp': ISTPAgent,
            'isfp': ISFPAgent,
            'estp': ESTPAgent,
            'esfp': ESFPAgent,
        }

class AgentsManager:
    """
    智能体管理器，用于创建和管理不同类型的智能体。
    """
    def __init__(self,workspace_root: str = "workspace",
                 log_level: str ="DEBUG"
                 ):
        self.workspace_root = workspace_root
        self.log_level = log_level
        
        self.message_manager = MessageManager(f'./{self.workspace_root}/messages')
        self.speaking_queue = SpeakingQueue()


        
        
    def create_mbti_agent(self,agent_id: str,task_id:str = "default") -> BaseAgent:
        """
        创建一个 agent_id 类型的智能体。
        Args:
            agent_id: 智能体的标识符。
            task_id: 任务标识符。
        Returns:
            Agent 实例。
        """
        # 设置任务级日志
        self.task_logger = setup_logger(
            task_id=task_id,
            workspace_root=self.workspace_root,
            log_level=self.log_level
        )
        
        # 初始化知识库
        self.init_knowledge_base()
        
        agent_type = agent_id[:4]  # 提取前缀，如 "infj"。默认这里一定是前4个字符。
        agent_class = AGENT_CLASS_MAP.get(agent_type)
        if not agent_class:
            raise ValueError(f"未知的智能体类型: {agent_type}")

        config = config_loader.load_config(agent_id=agent_id)
        system_config = config_loader.load_system_config()
        agent = agent_class(config, self.message_manager, self.speaking_queue, self.task_logger, system_config=system_config)
        agent.current_task_id = task_id
        return agent

    def get_agent_class(self,agent_type: str) -> BaseAgent:
        """
        根据智能体类型获取对应的智能体类。
        Args:
            agent_type: 智能体类型标识符。
        Returns:
            Agent 类。
        """
        agent_type = agent_type[:4].lower()  # 提取前缀并转换为小写。前四个字符一定是类型标识符。
        agent_class = AGENT_CLASS_MAP.get(agent_type)
        if not agent_class:
            raise ValueError(f"未知的智能体类型: {agent_type}")
        return agent_class

    def init_knowledge_base(self):
        """
        初始化知识库目录结构。
        """
        system_config = config_loader.load_config() # 加载系统配置
        
        embedding_client = create_llm_client(config=system_config, embedding=True)
        embedding_model = system_config.embedding.model_name
        dimension = system_config.embedding.dimension if system_config.embedding.dimension else 1536 # 默认text-embedding-3-small的维度
        init_knowledge_manager(
            workspace_root=self.workspace_root,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
            dimension=dimension,
            logger=self.task_logger
        )