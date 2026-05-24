'''
定义 DyCo 智能体系统入口，封装 discussion_workflow 并提供默认参数。
MBTI 仅作为可解释的角色先验测试床。
'''
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from configs import ConfigLoader, config_loader
from src.agents import AgentsManager
from src.agents.base_agent import BaseAgent
from src.communication.message_manager import MessageManager
from src.communication.speaking_queue import SpeakingQueue
from src.communication.team_manager import TeamManager
from src.logger import setup_logger
from src.workflows.discussion_workflow import FourRoundDiscussionWorkflow
from src.tools.system.knowledgemanager import init_knowledge_manager
from src.utils.llm_client import create_llm_client

class MBTIAgentSystem:
    """DyCo 智能体系统入口（MBTI 作为角色先验测试床）"""

    def __init__(
        self,
        config_dir: Optional[str] = "configs",
        *,
        task_id: Optional[str] = None,
        agent_ids: Optional[Iterable[str]] = None,
        reuse_global_loader: bool = False,
    ) -> None:
        """初始化系统组件并加载智能体"""
        self.model_name = "MBTI_MAS" # 默认模型名，基于 task_id 创建独立知识库路径
        self.config_loader = self._init_config_loader(config_dir, reuse_global_loader)
        self.system_config = self.config_loader.load_system_config()

        self.task_id = task_id or self._generate_task_id()
        self.knowledge_base_path = f"{self.model_name}/{self.task_id}"  
        log_level = getattr(logging, self.system_config.log_level.upper(), logging.INFO)
        self.task_logger = setup_logger(
            task_id=self.task_id,
            workspace_root=self.system_config.workspace_root,
            model_name=self.model_name,
            log_level=log_level,
        )
        self.task_logger.info("✅ 系统日志器已就绪")

        self.agent_ids = list(agent_ids) if agent_ids else self.config_loader.get_agent_ids()
        if not self.agent_ids:
            raise ValueError("系统配置中未找到任何智能体 ID")
        self.agentsmanager = AgentsManager()
        
        self.task_logger.debug("加载智能体配置文件…")
        self.agent_configs = [
            self.config_loader.load_agent_config(agent_id=agent_id) for agent_id in self.agent_ids
        ]
        self.task_logger.info("✅ 已加载 %d 个智能体配置", len(self.agent_configs))

        # 初始化通信组件
        self.task_logger.debug("初始化通信组件…")
        storage_root = Path(self.system_config.workspace_root)
        storage_root.mkdir(parents=True, exist_ok=True)
        message_storage_path = storage_root / "messages" / self.model_name
        message_storage_path.mkdir(parents=True, exist_ok=True)

        self.message_manager = MessageManager(
            storage_path=str(message_storage_path),
            storage_type=self.system_config.message_storage_type,
        )
        self.speaking_queue = SpeakingQueue()
        self.team_manager = TeamManager(storage_path=str(message_storage_path))
        self.task_logger.info("✅ 通信组件初始化完成")

        # 初始化知识库（必须在智能体初始化之前）
        self.task_logger.debug("初始化知识库管理器...")
        self.knowledge_manager = self.init_knowledge_base()
        self.task_logger.info("✅ 知识库管理器初始化完成")

        # 初始化智能体实例
        self.agents: List[BaseAgent] = []
        for agent_config in self.agent_configs:
            agent_class = self.agentsmanager.get_agent_class(agent_config.agent_id)
            if not agent_class:
                raise ValueError(f"未知的智能体 ID: {agent_config.agent_id}")

            agent = agent_class(
                agent_config,
                self.message_manager,
                self.speaking_queue,
                self.task_logger,
                system_config=self.system_config,
                knowledge_manager=self.knowledge_manager,  # 传入知识库管理器
            )
            self.agents.append(agent)
            self.task_logger.debug("智能体 %s 初始化完成", agent.agent_id)

        self.task_logger.info("✅ 成功初始化 %d 个智能体", len(self.agents))
            
        self.workflow = FourRoundDiscussionWorkflow(
            agents=self.agents,
            message_manager=self.message_manager,
            speaking_queue=self.speaking_queue,
            team_manager=self.team_manager,
            task_id=self.task_id,
            model_name=self.model_name,
            logger=self.task_logger,
            system_config=self.system_config,
        )
        self.task_logger.info("✅ 四轮讨论工作流已创建")

    def solve_task(self, task: str, initial_agent_id: Optional[str] = None):
        """
        运行工作流以解决任务
        
        Args:
            task: 任务描述
            initial_agent_id: 初始智能体ID（可选，默认使用第一个智能体）
            
        Returns:
            Dict[str, Any]: 工作流执行结果字典，包含：
                - success: bool - 执行是否成功
                - final_answer: str - 最终答案
                - workflow_path: str - 执行路径
                - total_rounds: int - 执行轮数
                - 其他详细结果字段...
        """
        if initial_agent_id is None and self.agents:
            initial_agent_id = self.agents[0].agent_id
        elif initial_agent_id and initial_agent_id not in {agent.agent_id for agent in self.agents}:
            raise ValueError(f"Agent {initial_agent_id} not found")

        self.task_logger.info("🚀 开始执行任务: %s", task)
        workflow_result = self.workflow.run(task, initial_agent_id)
        
        # 记录工作流执行结果
        if workflow_result.get('success'):
            self.task_logger.info("✅ 任务完成成功")
            self.task_logger.info(f"   执行路径: {workflow_result.get('workflow_path')}")
            self.task_logger.info(f"   执行轮数: {workflow_result.get('total_rounds')}")
            self.task_logger.info(f"   最终代表: {workflow_result.get('final_representative')}")
            self.task_logger.info(f"   答案长度: {len(workflow_result.get('final_answer', ''))} 字符")
        else:
            self.task_logger.error(f"❌ 任务执行失败: {workflow_result.get('error_message')}")
        
        return workflow_result

    def get_agent_by_id(self, agent_id: str) -> BaseAgent:
        """根据 ID 查找智能体"""

        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise ValueError(f"Agent {agent_id} not found")

    def list_agents(self) -> List[str]:
        """返回已加载的智能体 ID 列表"""

        return [agent.agent_id for agent in self.agents]

    def _init_config_loader(
        self,
        config_dir: Optional[str],
        reuse_global_loader: bool,
    ) -> ConfigLoader:
        """根据路径初始化配置加载器"""

        if reuse_global_loader or not config_dir:
            return config_loader

        config_path = Path(config_dir)
        if not config_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            config_path = (project_root / config_dir).resolve()

        return ConfigLoader(config_path)

    @staticmethod
    def _generate_task_id() -> str:
        """生成默认任务 ID"""

        return f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def init_knowledge_base(self):
        """
        初始化知识库目录结构。
        基于 task_id 创建独立的知识库，实现不同任务间的知识隔离。
        
        Returns:
            KnowledgeManager: 知识库管理器实例
        """
        embedding_client = create_llm_client(config=self.system_config, embedding=True)
        embedding_model = self.system_config.embedding.model_name
        dimension = self.system_config.embedding.dimension if self.system_config.embedding.dimension else 1536 # 默认text-embedding-3-small的维度
        
        # 传递 task_id 作为 knowledge_base_id，实现知识库隔离
        self.task_logger.info(f"📚 初始化任务专属知识库: {self.task_id}")
        
        # init_knowledge_manager 会设置全局知识库管理器
        init_knowledge_manager(
            workspace_root=self.system_config.workspace_root,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
            dimension=dimension,
            knowledge_base_id=self.knowledge_base_path,  # 传递 task_id 实现知识库隔离
            logger=self.task_logger
        )
        
        # 返回全局知识库管理器实例
        from src.tools.system.knowledgemanager import _global_knowledge_manager
        return _global_knowledge_manager
