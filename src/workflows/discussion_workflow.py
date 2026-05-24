"""
工作流模块 - 实现含评测与动态组队的四轮讨论流程
"""
from typing import List, Dict, Any, Optional, Set
import json
import logging
import random
import inspect
from datetime import datetime

from ..models.message import RoundStage, MessageType
from ..models.team import Team
from ..agents.base_agent import BaseAgent
from ..communication.message_manager import MessageManager
from ..communication.speaking_queue import SpeakingQueue
from ..communication.team_manager import TeamManager
from configs import config_loader
from ..utils.llm_client import create_llm_client,call_llm
from ..prompts import prompt_loader
from ..utils.helpers import get_tag_content
from src.utils.discussion_summary_manager import DiscussionSummaryManager

class FourRoundDiscussionWorkflow:
    """四轮讨论工作流（含评测、组队与代表讨论阶段）"""
    
    def __init__(self, 
                 agents: List[BaseAgent],
                 message_manager: MessageManager,
                 speaking_queue: SpeakingQueue,
                 team_manager: TeamManager,
                 task_id: str,
                 model_name: str = "MBTI_MAS",
                 logger: Optional[logging.Logger] = None,
                 system_config: Optional[Dict[str, Any]] = None):
        """
        初始化工作流
        
        Args:
            agents: 智能体列表
            message_manager: 消息管理器
            speaking_queue: 发言队列
            team_manager: 团队管理器
            task_id: 任务ID
            logger: 日志记录器
            system_config: 系统配置(包含model配置)
        """
        self.agents = agents

        self.message_manager = message_manager
        self.speaking_queue = speaking_queue
        self.team_manager = team_manager
        self.task_id = task_id
        self._init_agents()
        if logger is None:
            raise ValueError("Logger must be provided")
        self.logger = logger
        self.system_config = system_config or {}
        
        # 初始化LLM客户端
        self.llm_client = None
        self._init_llm_client()
        
        # 初始化讨论摘要管理器
        # 如果 system_config 是 SystemConfig 对象，直接访问属性；如果是字典，使用 get
        if hasattr(self.system_config, 'workspace_root'):
            workspace_root = self.system_config.workspace_root
        elif isinstance(self.system_config, dict):
            workspace_root = self.system_config.get('workspace_root', 'workspace')
        else:
            workspace_root = 'workspace'
        
        self.discussion_summary_manager = DiscussionSummaryManager(
            task_id=task_id,
            workspace_root=workspace_root,
            model_name=model_name
        )
        
        # 为所有智能体注入讨论摘要管理器
        for agent in self.agents:
            agent.discussion_summary_manager = self.discussion_summary_manager
        
        self.logger.info(f"初始化讨论工作流")
        self.logger.info(f"  任务ID: {task_id}")
        self.logger.info(f"  参与智能体数: {len(agents)}")
        self.logger.debug(f"  智能体列表: {[a.agent_id for a in agents]}")
        self.logger.info(f"✅ 讨论摘要管理器已初始化")
        
    def run(self, task: str, initial_agent_id: str) -> Dict[str, Any]:
        """
        运行完整的四轮讨论流程
        
        Args:
            task: 任务描述
            initial_agent_id: 初始解决任务的智能体ID
            
        Returns:
            Dict[str, Any]: 工作流执行结果，包含以下字段：
                - success: bool - 执行是否成功
                - task: str - 任务描述
                - initial_agent_id: str - 初始智能体ID
                - workflow_path: str - 执行路径（'direct_approval' 或 'full_discussion'）
                - initial_solution_result: Dict - 初始方案结果
                - evaluation_result: Dict - 评测结果
                - round1_result: Dict - 第一轮讨论结果
                - round2_preference_result: Dict - 第二轮意愿收集结果
                - round2_clustering_result: Dict - 第二轮分组结果
                - round3_result: Dict - 第三轮小组讨论结果
                - round4_result: Dict - 第四轮代表讨论结果
                - final_answer: str - 最终答案
                - final_representative: str - 最终撰写代表
                - total_rounds: int - 实际执行轮数
                - timestamp: str - 完成时间戳
                - error_message: str - 错误信息（如有）
        """
        from datetime import datetime
        
        # 初始化结果收集器
        workflow_result = {
            'success': False,
            'task': task,
            'initial_agent_id': initial_agent_id,
            'workflow_path': '',
            'initial_solution_result': {},
            'evaluation_result': {},
            'round1_result': {},
            'round2_preference_result': {},
            'round2_clustering_result': {},
            'round3_result': {},
            'round4_result': {},
            'final_answer': '',
            'final_representative': '',
            'total_rounds': 0,
            'timestamp': '',
            'error_message': ''
        }
        
        try:
            # 设置任务到讨论摘要管理器
            self.discussion_summary_manager.set_task(task)
            self.logger.info(f"✓ 已设置任务到讨论摘要管理器")
            
            self.logger.debug("="*80)
            self.logger.info("🚀 开始MBTI讨论流程")
            self.logger.debug(f"任务: {task}")
            self.logger.info(f"初始智能体: {initial_agent_id}")
            self.logger.debug("="*80)

            #########初始方案阶段########
            # 初始解决阶段。生成一个初始解决方案。
            self.logger.debug("\n" + "="*80)
            self.logger.info(f"📝 阶段{RoundStage.INITIAL_SOLUTION} : 初始解决")
            initial_result = self.initial_solution_phase(task, initial_agent_id)
            workflow_result['initial_solution_result'] = initial_result
            
            # 检查是否成功生成
            if not initial_result['success']:
                error_msg = f"初始方案生成失败: {initial_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                workflow_result['error_message'] = error_msg
                workflow_result['timestamp'] = datetime.now().isoformat()
                return workflow_result
            
            initial_solution = initial_result['solution']
            self.logger.debug(f"顺利生成初始解决方案，长度为{len(initial_solution)}")
            self.logger.debug("="*80)
            
            ## 每个智能体对初始方案进行评测
            evaluation_result = self.evaluate_initial_solution(task, initial_solution, initial_agent_id)
            workflow_result['evaluation_result'] = evaluation_result
            
            # 检查评测是否成功
            if not evaluation_result['success']:
                self.logger.warning(f"⚠️ 评测阶段出现问题: {evaluation_result['error_message']}")
                # 如果没有评审者或评测失败，默认进入讨论流程
                evaluations = {}
            else:
                evaluations = evaluation_result['evaluations']
                self.logger.info(f"📊 评测统计: 通过率 {evaluation_result['approval_rate']:.1%}，平均信心 {evaluation_result['average_confidence']:.1f}/100")
            
            # 判断该方案是否直接通过
            transition_result = self._transition_from_initial_solution_2_round_1_discussion(
                task, initial_solution, evaluation_result
            )
            self.logger.debug("="*80)
            
            # 检查过渡判断结果
            if transition_result.get('approved', False):
                self.logger.info("🎉 初始方案评估通过，直接采用该方案作为最终答案")
                self.logger.info(f"   决策理由: {transition_result.get('decision_reason', '未提供')}")
                self.logger.debug("="*80)
                
                # 直接通过路径
                workflow_result['success'] = True
                workflow_result['workflow_path'] = 'direct_approval'
                workflow_result['final_answer'] = initial_solution
                workflow_result['final_representative'] = initial_agent_id
                workflow_result['total_rounds'] = 0
                workflow_result['timestamp'] = datetime.now().isoformat()
                return workflow_result
            
            self.logger.info("➡️ 初始方案未通过，进入4轮讨论流程")
            self.logger.info(f"   决策理由: {transition_result.get('decision_reason', '未提供')}")
            self.logger.debug("="*80)
            workflow_result['workflow_path'] = 'full_discussion'
            
            
            ## 没有通过
            # 第一轮: 全体讨论 - 分析任务
            self.logger.info("\n" + "="*80)
            self.logger.info("💭 第一轮: 全体讨论 - 分析任务")
            self.logger.info("="*80)
            discussion_result = self.round_1_all_discussion(task)
            workflow_result['round1_result'] = discussion_result
            
            # 检查讨论是否成功
            if not discussion_result['success']:
                self.logger.warning(f"⚠️ 第一轮讨论出现问题: {discussion_result['error_message']}")
                # 如果讨论失败，使用空字典作为分析结果
                analyses = {}
            else:
                analyses = discussion_result['analyses']
                self.logger.info(
                    f"📊 讨论统计: 成功分析 {discussion_result['successful_analyses']}/{discussion_result['participant_count']}"
                )
            
            # 第二轮: 组队
            self.logger.info("\n" + "="*80)
            self.logger.info("👥 第二轮: 组队")
            self.logger.info("="*80)
            preference_result = self.round_2_team_formation_decide_team_preference(task, analyses)
            workflow_result['round2_preference_result'] = preference_result
            
            # 检查组队意愿收集是否成功
            if not preference_result['success']:
                self.logger.warning(f"⚠️ 组队意愿收集出现问题: {preference_result['error_message']}")
                # 即使部分失败，也继续使用已收集的意愿
                agents_team_preference = preference_result['preferences']
            else:
                agents_team_preference = preference_result['preferences']
                self.logger.info(
                    f"📊 组队意愿统计: 成功表态 {preference_result['successful_count']}/{preference_result['participant_count']}"
                )
            n_teams = 2 # TODO 放到配置中
            # 调用过渡函数进行分组
            clustering_result = self._transition_from_round_2_team_formation_2_round_3_team_discussion(task, agents_team_preference,n_teams=n_teams)
            workflow_result['round2_clustering_result'] = clustering_result
            
            # 检查分组是否成功
            if not clustering_result['success']:
                error_msg = f"组队分组失败: {clustering_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                workflow_result['error_message'] = error_msg
                workflow_result['timestamp'] = datetime.now().isoformat()
                return workflow_result
            
            teams = clustering_result['teams']
            self.logger.info(f"✓ 组队分组成功，形成 {clustering_result['team_count']} 个团队")
            self.logger.debug(f"  聚类方法: {clustering_result['clustering_method']}")
            
            # 第三轮: 小组讨论
            self.logger.info("\n" + "="*80)
            self.logger.info("🗣️ 第三轮: 小组讨论")
            self.logger.info("="*80)
            round3_result = self.round_3_team_discussion(task, teams)
            workflow_result['round3_result'] = round3_result
            
            # 检查第三轮是否成功
            if not round3_result['success']:
                error_msg = f"第三轮讨论失败: {round3_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                workflow_result['error_message'] = error_msg
                workflow_result['timestamp'] = datetime.now().isoformat()
                return workflow_result
            
            representation_agents = round3_result['representatives']
            self.logger.info(f"✓ 第三轮讨论完成，选出 {len(representation_agents)} 个团队代表")
            
            # 第四轮: 代表讨论
            self.logger.info("\n" + "="*80)
            self.logger.info("🎯 第四轮: 代表讨论")
            self.logger.info("="*80)
            round4_result = self.round_4_final_discussion(task, representation_agents)
            workflow_result['round4_result'] = round4_result
            
            # 检查第四轮是否成功
            if not round4_result['success']:
                error_msg = f"第四轮讨论失败: {round4_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                workflow_result['error_message'] = error_msg
                workflow_result['timestamp'] = datetime.now().isoformat()
                return workflow_result
            
            final_answer = round4_result['final_answer']
            final_representative = round4_result['final_representative']
            self.logger.info(f"✓ 第四轮讨论完成，最终撰写代表: {final_representative}")
            
            # 设置最终结果
            workflow_result['success'] = True
            workflow_result['final_answer'] = final_answer
            workflow_result['final_representative'] = final_representative
            workflow_result['total_rounds'] = 4
            workflow_result['timestamp'] = datetime.now().isoformat()
               
            self.logger.info("\n" + "="*80)
            self.logger.info("✅ 四轮讨论流程完成")
            self.logger.info("="*80)
            
            return workflow_result
            
        except Exception as e:
            error_msg = f"讨论流程执行异常: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.logger.exception(e)
            workflow_result['error_message'] = error_msg
            workflow_result['timestamp'] = datetime.now().isoformat()
            return workflow_result

    def _init_agents(self):
        """初始化智能体"""
        for agent in self.agents:
            agent.current_task_id = self.task_id
  
    def _init_llm_client(self):
        """初始化裁决LLM客户端"""
        try:
            system_config = config_loader.load_config() # 默认加载系统配置
            self.llm_client = create_llm_client(system_config.model)
            self.logger.info("✓ LLM客户端初始化成功")                
        except Exception as e:
            self.logger.error(f"❌ 初始化裁决LLM客户端失败: {e}", exc_info=True)
            self.llm_client = None

    def _get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """根据ID获取智能体"""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def initial_solution_phase(self, task: str, agent_id: str) -> Dict[str, Any]:
        """
        初始解决阶段 - 由指定智能体生成初步解决方案
        
        执行流程：
        1. 验证智能体存在性
        2. 创建初始团队（仅包含该智能体）
        3. 调用智能体的 generate 方法生成结构化方案
        4. 记录发言到消息系统
        5. 返回结构化结果
        
        Args:
            task: 任务描述
            agent_id: 负责初始解决的智能体ID
            
        Returns:
            Dict[str, Any]: 初始化方案字典，包含以下字段：
            {
                'success': bool,  # 是否成功生成
                'agent_id': str,  # 生成方案的智能体ID
                'team_id': str,  # 团队ID
                'solution': str,  # 生成的解决方案文本（主要内容）
                'generate_result': Dict[str, Any],  # generate方法的完整返回值
                'timestamp': str,  # 生成时间戳
                'error_message': str  # 错误信息（如果失败）
            }
        """
        function_name = inspect.currentframe().f_code.co_name
        logger_prefix = f"【{function_name}】"
        
        self.logger.info(f"{logger_prefix} ==================== 初始解决阶段开始 ====================")
        self.logger.info(f"{logger_prefix} 任务长度: {len(task)} 字符")
        self.logger.info(f"{logger_prefix} 初始智能体: {agent_id}")
        
        # 初始化返回结果
        result = {
            'success': False,
            'agent_id': agent_id,
            'team_id': '',
            'solution': '',
            'generate_result': {},
            'timestamp': datetime.now().isoformat(),
            'error_message': ''
        }
        
        try:
            # ========== 1. 验证智能体存在性 ==========
            self.logger.debug(f"{logger_prefix} 步骤1: 查找智能体 {agent_id}...")
            agent = self._get_agent_by_id(agent_id)
            if not agent:
                error_msg = f"智能体 {agent_id} 不存在"
                self.logger.error(f"{logger_prefix} ❌ {error_msg}")
                result['error_message'] = error_msg
                raise ValueError(error_msg)
            
            self.logger.info(f"{logger_prefix} ✓ 智能体验证成功: {agent.agent_name} (MBTI: {agent.mbti_type})")
            
            # ========== 2. 创建初始团队 ==========
            self.logger.debug(f"{logger_prefix} 步骤2: 创建初始团队（成员: {agent_id}）...")
            team_id = Team.generate_team_id([agent_id])
            result['team_id'] = team_id
            
            self.logger.info(f"{logger_prefix} ✓ 团队ID生成成功: {team_id}")
            
            # ========== 3. 设置当前任务ID ==========
            self.logger.debug(f"{logger_prefix} 步骤3: 设置任务上下文...")
            agent.current_task_id = self.task_id
            self.logger.debug(f"{logger_prefix} ✓ 任务ID已设置: {self.task_id}")
            
            # ========== 4. 调用 generate 生成解决方案 ==========
            self.logger.info(f"{logger_prefix} 步骤4: 调用 {agent_id}.generate() 生成解决方案...")
            self.logger.debug(f"{logger_prefix}   任务内容: {task[:200]}{'...' if len(task) > 200 else ''}")
            
            generate_result = agent.generate(task)
            result['generate_result'] = generate_result
            
            # 从 generate 返回的字典中提取主要内容
            if isinstance(generate_result, dict):
                # 优先使用 main_solution（新版格式）
                solution_text = generate_result.get('main_solution', '')
                
                # 如果 main_solution 为空，尝试直接使用 content
                if not solution_text and 'content' in generate_result:
                    # content 可能是字典或字符串
                    content_field = generate_result.get('content', {})
                    if isinstance(content_field, dict):
                        # 从 content 字典中提取 main_content 或其他字段
                        solution_text = content_field.get('main_content', '')
                        if not solution_text:
                            # 尝试拼接所有内容字段
                            solution_text = '\n\n'.join([
                                f"【{k}】\n{v}" for k, v in content_field.items() if v
                            ])
                    elif isinstance(content_field, str):
                        solution_text = content_field
                
                knowledge_count = generate_result.get('knowledge_points_count', 0)
                
                self.logger.info(f"{logger_prefix} ✓ 方案生成成功")
                self.logger.info(f"{logger_prefix}   解决方案长度: {len(solution_text)} 字符")
                self.logger.info(f"{logger_prefix}   知识点存储数: {knowledge_count} 个")
                
                if 'thinking_summary' in generate_result:
                    thinking = generate_result.get('thinking_summary', '')
                    if thinking:
                        self.logger.debug(f"{logger_prefix}   思考摘要: {thinking[:100]}...")
                
                # 如果解析后仍然为空，记录警告并使用原始响应
                if not solution_text:
                    self.logger.warning(f"{logger_prefix} ⚠️ 未能从 generate_result 提取有效内容，尝试使用原始响应")
                    # 检查是否有原始响应文本
                    if 'raw_response' in generate_result:
                        solution_text = generate_result['raw_response']
                    else:
                        # 作为最后手段，将整个字典转为字符串
                        solution_text = str(generate_result)
                        self.logger.warning(f"{logger_prefix} ⚠️ 最终回退方案：使用 generate_result 字符串表示")
                
                result['solution'] = solution_text
            else:
                # 兜底：如果返回的不是字典，直接使用返回值作为解决方案
                self.logger.warning(f"{logger_prefix} ⚠️ generate() 返回值非字典类型，使用原始返回值")
                solution_text = str(generate_result)
                result['solution'] = solution_text
            
            # ========== 5. 记录发言到消息系统 ==========
            self.logger.debug(f"{logger_prefix} 步骤5: 记录发言到消息系统...")
            agent.speak(
                task_id=self.task_id,
                team_id=team_id,
                content=solution_text,
                message_type=MessageType.SOLUTION,
                round_stage=RoundStage.INITIAL_SOLUTION
            )
            self.logger.info(f"{logger_prefix} ✓ 发言已记录 (消息类型: SOLUTION)")
            
            # ========== 6. 标记成功 ==========
            result['success'] = True
            self.logger.info(f"{logger_prefix} ==================== 初始解决阶段完成 ====================")
            
        except Exception as e:
            error_msg = f"初始解决阶段失败: {str(e)}"
            self.logger.error(f"{logger_prefix} ❌ {error_msg}", exc_info=True)
            result['error_message'] = error_msg
            result['success'] = False
            
            # 即使失败也返回结果字典，而不是抛出异常
            # 这样上层调用者可以根据 success 字段判断并处理
        
        return result

    def evaluate_initial_solution(self, task: str, initial_solution: str, initial_agent_id: str) -> Dict[str, Any]:
        """
        初始方案评测阶段 - 其他智能体对初步解决方案进行结构化评测
        
        执行流程：
        1. 筛选评审智能体（排除提案者）
        2. 创建评测团队
        3. 逐个智能体调用 evaluate_solution 方法
        4. 记录评测结果到消息系统
        5. 统计评测结果并返回结构化数据
        
        Args:
            task: 任务描述
            initial_solution: 初始解决方案
            initial_agent_id: 提出初始方案的智能体ID
            
        Returns:
            Dict[str, Any]: 评测结果字典，包含以下字段：
            {
                'success': bool,  # 是否成功完成评测
                'evaluations': Dict[str, Dict],  # 评测结果 {agent_id: evaluation_result}
                'evaluator_count': int,  # 参与评审的智能体数量
                'approval_count': int,  # 通过的评审数量
                'approval_rate': float,  # 通过率（0-1）
                'average_confidence': float,  # 平均信心程度（0-100）
                'team_id': str,  # 评测团队ID
                'timestamp': str,  # 评测时间戳
                'error_message': str  # 错误信息（如果失败）
            }
        """
        function_name = inspect.currentframe().f_code.co_name
        logger_prefix = f"【{function_name}】"
        
        self.logger.info(f"{logger_prefix} ==================== 初始方案评测阶段开始 ====================")
        self.logger.info(f"{logger_prefix} 任务长度: {len(task)} 字符")
        self.logger.info(f"{logger_prefix} 方案长度: {len(initial_solution)} 字符")
        self.logger.info(f"{logger_prefix} 提案智能体: {initial_agent_id}")
        
        # 初始化返回结果
        result = {
            'success': False,
            'evaluations': {},
            'evaluator_count': 0,
            'approval_count': 0,
            'approval_rate': 0.0,
            'average_confidence': 0.0,
            'team_id': '',
            'timestamp': datetime.now().isoformat(),
            'error_message': ''
        }
        
        try:
            # ========== 1. 筛选评审智能体 ==========
            self.logger.info(f"{logger_prefix} 步骤1: 筛选评审智能体...")
            # 找出所有需要评测的智能体（排除提案者）
            all_evaluators = [agent for agent in self.agents if agent.agent_id != initial_agent_id]
            self.logger.debug(f"{logger_prefix}   候选评审智能体数量: {len(all_evaluators)}")
            
            # 随机选择评审智能体
            max_evaluators = getattr(self.system_config, 'max_evaluation_agents', 3)
            if len(all_evaluators) <= max_evaluators:
                # 如果候选者不多，全部参与评审
                evaluators = all_evaluators
                self.logger.debug(f"{logger_prefix}   候选者较少，全部参与评审")
            else:
                # 随机选择指定数量的评审者
                import random
                random_indices = random.sample(range(len(all_evaluators)), max_evaluators)
                evaluators = [all_evaluators[i] for i in random_indices]
                self.logger.debug(f"{logger_prefix}   随机选择 {max_evaluators} 位评审智能体")
            
            if not evaluators:
                warning_msg = "没有可用的评审智能体，跳过评测"
                self.logger.warning(f"{logger_prefix} ⚠️ {warning_msg}")
                result['error_message'] = warning_msg
                result['success'] = True  # 虽然没有评审者，但流程正常
                return result
            
            result['evaluator_count'] = len(evaluators)
            self.logger.info(f"{logger_prefix} ✓ 筛选完成，共 {len(evaluators)} 位评审智能体")
            self.logger.debug(f"{logger_prefix}   评审名单: {[e.agent_id for e in evaluators]}")
            
            # ========== 2. 创建评测团队 ==========
            self.logger.info(f"{logger_prefix} 步骤2: 创建评测团队...")
            all_agent_ids = [agent.agent_id for agent in self.agents]
            evaluation_team_id = Team.generate_team_id(all_agent_ids)
            result['team_id'] = evaluation_team_id
            
            if not self.team_manager.get_team(self.task_id, evaluation_team_id):
                self.team_manager.create_team(
                    self.task_id,
                    all_agent_ids,
                    RoundStage.INITIAL_SOLUTION
                )
                self.logger.debug(f"{logger_prefix}   新建评测团队")
            else:
                self.logger.debug(f"{logger_prefix}   使用已有团队")
            
            self.logger.info(f"{logger_prefix} ✓ 团队ID: {evaluation_team_id}")
            
            # ========== 3. 逐个智能体进行评测 ==========
            self.logger.info(f"{logger_prefix} 步骤3: 执行评测（共 {len(evaluators)} 位评审）...")
            evaluations = {}
            approval_count = 0
            total_confidence = 0.0
            successful_evaluations = 0
            
            for idx, agent in enumerate(evaluators, 1):
                self.logger.debug(f"{logger_prefix}   [{idx}/{len(evaluators)}] {agent.agent_id} 开始评测...")
                
                try:
                    # 设置任务上下文
                    agent.current_task_id = self.task_id
                    
                    # 调用智能体的 evaluate_solution 方法
                    evaluation_result = agent.evaluate_solution(task, initial_solution)
                    
                    # 验证返回值格式
                    if not isinstance(evaluation_result, dict):
                        self.logger.warning(f"{logger_prefix}   ⚠️ {agent.agent_id} 返回值非字典，尝试转换")
                        evaluation_result = {
                            'approved': False,
                            'approval_status': 'error',
                            'confidence_level': 0,
                            'summary': str(evaluation_result),
                            'evaluation_points_count': 0
                        }
                    
                    evaluations[agent.agent_id] = evaluation_result
                    
                    # 统计通过情况
                    if evaluation_result.get('approved', False):
                        approval_count += 1
                    
                    # 累计信心程度
                    confidence = evaluation_result.get('confidence_level', 0)
                    total_confidence += confidence
                    successful_evaluations += 1
                    
                    self.logger.debug(f"{logger_prefix}   ✓ {agent.agent_id} 评测完成")
                    self.logger.debug(f"{logger_prefix}     - 通过: {evaluation_result.get('approved', False)}")
                    self.logger.debug(f"{logger_prefix}     - 状态: {evaluation_result.get('approval_status', 'unknown')}")
                    self.logger.debug(f"{logger_prefix}     - 信心: {confidence}/100")
                    
                    # ========== 4. 记录评测发言到消息系统 ==========
                    # 构建评测摘要（用于消息记录）
                    evaluation_summary = (
                        f"评审结果：{'✓ 通过' if evaluation_result.get('approved') else '✗ 不通过'}\n"
                        f"状态：{evaluation_result.get('approval_status', 'unknown')}\n"
                        f"信心程度：{confidence}/100\n"
                        f"总结：{evaluation_result.get('summary', '无')}\n"
                        f"评价点数量：{evaluation_result.get('evaluation_points_count', 0)}"
                    )
                    
                    agent.speak(
                        task_id=self.task_id,
                        team_id=evaluation_team_id,
                        content=evaluation_summary,
                        message_type=MessageType.EVALUATION,
                        round_stage=RoundStage.INITIAL_SOLUTION
                    )
                    self.logger.debug(f"{logger_prefix}   ✓ 评测结果已记录到消息系统")
                    
                except Exception as exc:
                    error_msg = f"评测失败: {str(exc)}"
                    self.logger.error(f"{logger_prefix}   ❌ {agent.agent_id} {error_msg}", exc_info=True)
                    
                    # 记录错误到评测结果
                    evaluations[agent.agent_id] = {
                        'approved': False,
                        'approval_status': 'error',
                        'confidence_level': 0,
                        'summary': error_msg,
                        'evaluation_points_count': 0,
                        'error': str(exc)
                    }
            
            result['evaluations'] = evaluations
            self.logger.info(f"{logger_prefix} ✓ 所有评测完成（成功 {successful_evaluations}/{len(evaluators)}）")
            
            # ========== 5. 统计评测结果 ==========
            self.logger.info(f"{logger_prefix} 步骤4: 统计评测结果...")
            result['approval_count'] = approval_count
            result['approval_rate'] = approval_count / len(evaluators) if evaluators else 0.0
            result['average_confidence'] = total_confidence / successful_evaluations if successful_evaluations > 0 else 0.0
            
            self.logger.info(f"{logger_prefix}   通过数量: {approval_count}/{len(evaluators)}")
            self.logger.info(f"{logger_prefix}   通过率: {result['approval_rate']:.1%}")
            self.logger.info(f"{logger_prefix}   平均信心: {result['average_confidence']:.1f}/100")
            
            # ========== 6. 标记成功 ==========
            result['success'] = True
            self.logger.info(f"{logger_prefix} ==================== 初始方案评测阶段完成 ====================")
            
        except Exception as e:
            error_msg = f"初始方案评测阶段失败: {str(e)}"
            self.logger.error(f"{logger_prefix} ❌ {error_msg}", exc_info=True)
            result['error_message'] = error_msg
            result['success'] = False
        
        return result

    def _transition_from_initial_solution_2_round_1_discussion(
        self, 
        task: str, 
        initial_solution: str, 
        evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        从初始解决阶段过渡到第一轮讨论阶段 - 基于评测结果判断是否需要讨论
        
        执行流程：
        1. 提取评测结果中的 approval_rate
        2. 判断是否全体通过（approval_rate == 1.0）
        3. 记录判断结果到日志
        4. 返回结构化的过渡结果
        
        Args:
            task: 任务描述
            initial_solution: 初始解决方案
            evaluation_result: evaluate_initial_solution 的返回结果字典
            
        Returns:
            Dict[str, Any]: 过渡判断结果字典，包含以下字段：
            {
                'approved': bool,  # 是否通过（可以直接采纳初始方案）
                'approval_rate': float,  # 评测通过率（0-1）
                'evaluator_count': int,  # 参与评审的智能体数量
                'approval_count': int,  # 通过的评审数量
                'average_confidence': float,  # 平均信心程度（0-100）
                'decision_reason': str,  # 决策理由
                'timestamp': str,  # 判断时间戳
                'error_message': str  # 错误信息（如果失败）
            }
        """
        function_name = inspect.currentframe().f_code.co_name
        logger_prefix = f"【{function_name}】"
        
        self.logger.info(f"{logger_prefix} ==================== 过渡判断阶段开始 ====================")
        
        # 初始化返回结果
        result = {
            'approved': False,
            'approval_rate': 0.0,
            'evaluator_count': 0,
            'approval_count': 0,
            'average_confidence': 0.0,
            'decision_reason': '',
            'timestamp': datetime.now().isoformat(),
            'error_message': ''
        }
        
        try:
            # ========== 1. 提取评测结果数据 ==========
            self.logger.info(f"{logger_prefix} 步骤1: 提取评测结果数据...")
            
            # 验证 evaluation_result 格式
            if not isinstance(evaluation_result, dict):
                error_msg = f"evaluation_result 类型错误: 期望 dict，实际 {type(evaluation_result)}"
                self.logger.error(f"{logger_prefix} ❌ {error_msg}")
                result['error_message'] = error_msg
                return result
            
            # 提取关键指标
            approval_rate = evaluation_result.get('approval_rate', 0.0)
            evaluator_count = evaluation_result.get('evaluator_count', 0)
            approval_count = evaluation_result.get('approval_count', 0)
            average_confidence = evaluation_result.get('average_confidence', 0.0)
            
            result['approval_rate'] = approval_rate
            result['evaluator_count'] = evaluator_count
            result['approval_count'] = approval_count
            result['average_confidence'] = average_confidence
            
            self.logger.info(f"{logger_prefix} ✓ 数据提取完成")
            self.logger.debug(f"{logger_prefix}   评审智能体数量: {evaluator_count}")
            self.logger.debug(f"{logger_prefix}   通过数量: {approval_count}")
            self.logger.debug(f"{logger_prefix}   通过率: {approval_rate:.1%}")
            self.logger.debug(f"{logger_prefix}   平均信心: {average_confidence:.1f}/100")
            
            # ========== 2. 判断是否全体通过 ==========
            self.logger.info(f"{logger_prefix} 步骤2: 判断是否全体通过...")
            
            # 核心判断逻辑：approval_rate == 1.0 表示全体通过
            if approval_rate == 1.0:
                result['approved'] = True
                result['decision_reason'] = (
                    f"初始方案获得全体评审通过（{approval_count}/{evaluator_count}），"
                    f"平均信心程度 {average_confidence:.1f}/100，可直接采纳。"
                )
                self.logger.info(f"{logger_prefix} ✓ 判断结果: 通过")
                self.logger.info(f"{logger_prefix}   理由: 全体评审通过（{approval_count}/{evaluator_count}）")
                self.logger.info(f"{logger_prefix}   平均信心: {average_confidence:.1f}/100")
            else:
                result['approved'] = False
                reject_count = evaluator_count - approval_count
                result['decision_reason'] = (
                    f"初始方案未获得全体通过（{approval_count}/{evaluator_count}），"
                    f"有 {reject_count} 位评审提出异议，需要进入讨论阶段。"
                )
                self.logger.info(f"{logger_prefix} ✓ 判断结果: 不通过")
                self.logger.info(f"{logger_prefix}   理由: 存在异议（通过 {approval_count}/{evaluator_count}）")
                self.logger.info(f"{logger_prefix}   异议数量: {reject_count} 位")
                
            # ========== 3. 记录判断摘要到日志 ==========
            self.logger.info(f"{logger_prefix} 步骤3: 记录判断摘要...")
            self.logger.info(f"{logger_prefix}   决策: {'✅ 采纳初始方案' if result['approved'] else '❌ 进入讨论阶段'}")
            self.logger.debug(f"{logger_prefix}   决策理由: {result['decision_reason']}")
            
            # ========== 4. 完成过渡判断 ==========
            self.logger.info(f"{logger_prefix} ==================== 过渡判断阶段完成 ====================")
            
        except Exception as e:
            error_msg = f"过渡判断失败: {str(e)}"
            self.logger.error(f"{logger_prefix} ❌ {error_msg}", exc_info=True)
            result['error_message'] = error_msg
            result['approved'] = False
        
        return result

    def round_1_all_discussion(self, task: str) -> Dict[str, Any]:
        """
        第一轮全体讨论阶段 - 每个智能体分析任务，为组队做准备
        
        执行流程：
        1. 创建全员团队
        2. 重置发言序号
        3. 逐个智能体调用 analyze_task 方法
        4. 记录分析结果到消息系统
        5. 统计并返回结构化结果
        
        Args:
            task: 任务描述
            
        Returns:
            Dict[str, Any]: 讨论结果字典，包含以下字段：
            {
                'success': bool,  # 是否成功完成讨论
                'analyses': Dict[str, Dict],  # 分析结果 {agent_id: analysis_result}
                'participant_count': int,  # 参与讨论的智能体数量
                'successful_analyses': int,  # 成功完成分析的数量
                'team_id': str,  # 团队ID
                'timestamp': str,  # 讨论时间戳
                'error_message': str  # 错误信息（如果失败）
            }
        """
        function_name = inspect.currentframe().f_code.co_name
        logger_prefix = f"【{function_name}】"
        
        self.logger.info(f"{logger_prefix} ==================== 第一轮全体讨论阶段开始 ====================")
        self.logger.info(f"{logger_prefix} 任务长度: {len(task)} 字符")
        self.logger.info(f"{logger_prefix} 参与智能体数量: {len(self.agents)}")
        
        # 初始化返回结果
        result = {
            'success': False,
            'analyses': {},
            'participant_count': len(self.agents),
            'successful_analyses': 0,
            'team_id': '',
            'timestamp': datetime.now().isoformat(),
            'error_message': ''
        }
        
        try:
            # ========== 1. 创建全员团队 ==========
            self.logger.info(f"{logger_prefix} 步骤1: 创建全员团队...")
            all_agent_ids = [agent.agent_id for agent in self.agents]
            team_id = Team.generate_team_id(all_agent_ids)
            result['team_id'] = team_id
            
            # 创建团队记录
            self.team_manager.create_initial_team(self.task_id, all_agent_ids)
            
            self.logger.info(f"{logger_prefix} ✓ 团队创建成功")
            self.logger.debug(f"{logger_prefix}   团队ID: {team_id}")
            self.logger.debug(f"{logger_prefix}   成员: {', '.join(all_agent_ids)}")
            
            # ========== 2. 重置发言序号 ==========
            self.logger.info(f"{logger_prefix} 步骤2: 重置发言序号...")
            self.speaking_queue.reset_sequence()
            self.logger.debug(f"{logger_prefix} ✓ 发言序号已重置")
            
            # ========== 3. 逐个智能体分析任务 ==========
            self.logger.info(f"{logger_prefix} 步骤3: 执行任务分析（共 {len(self.agents)} 位智能体）...")
            analyses = {}
            successful_count = 0
            
            for idx, agent in enumerate(self.agents, 1):
                self.logger.debug(f"{logger_prefix}   [{idx}/{len(self.agents)}] {agent.agent_id} 开始分析...")
                
                try:
                    # 设置任务上下文
                    agent.current_task_id = self.task_id
                    
                    # 调用智能体的 analyze_task 方法
                    analysis_result = agent.analyze_task(task)
                    
                    # 验证返回值格式
                    if not isinstance(analysis_result, dict):
                        self.logger.warning(f"{logger_prefix}   ⚠️ {agent.agent_id} 返回值非字典，尝试转换")
                        analysis_result = {
                            'success': False,
                            'knowledge_points_count': 0,
                            'summary': str(analysis_result),
                            'error': '返回值格式错误'
                        }
                    
                    analyses[agent.agent_id] = analysis_result
                    
                    # 检查分析是否成功
                    if analysis_result.get('success', False):
                        successful_count += 1
                        
                        self.logger.debug(f"{logger_prefix}   ✓ {agent.agent_id} 分析完成")
                        self.logger.debug(f"{logger_prefix}     - 成功: {analysis_result.get('success', False)}")
                        self.logger.debug(f"{logger_prefix}     - 知识点: {analysis_result.get('knowledge_points_count', 0)} 个")
                        
                        # 提取摘要信息
                        summary = analysis_result.get('summary', '')
                        if summary:
                            summary_preview = summary[:100] if len(summary) > 100 else summary
                            self.logger.debug(f"{logger_prefix}     - 摘要: {summary_preview}...")
                    else:
                        self.logger.warning(f"{logger_prefix}   ⚠️ {agent.agent_id} 分析失败")
                    
                    # ========== 4. 记录分析结果到消息系统 ==========
                    # 构建分析摘要（用于消息记录）
                    if analysis_result.get('success', False):
                        analysis_summary = (
                            f"任务分析完成\n"
                            f"知识点数量：{analysis_result.get('knowledge_points_count', 0)}\n"
                            f"分析摘要：{analysis_result.get('summary', '无')}\n"
                        )
                    else:
                        analysis_summary = str(analysis_result)
                    
                    agent.speak(
                        task_id=self.task_id,
                        team_id=team_id,
                        content=analysis_summary,
                        message_type=MessageType.DISCUSSION,
                        round_stage=RoundStage.ROUND_1_DISCUSSION
                    )
                    self.logger.debug(f"{logger_prefix}   ✓ 分析结果已记录到消息系统")
                    
                except Exception as exc:
                    error_msg = f"分析失败: {str(exc)}"
                    self.logger.error(f"{logger_prefix}   ❌ {agent.agent_id} {error_msg}", exc_info=True)
                    
                    # 记录错误到分析结果
                    analyses[agent.agent_id] = {
                        'success': False,
                        'knowledge_points_count': 0,
                        'summary': error_msg,
                        'error': str(exc)
                    }
            
            result['analyses'] = analyses
            result['successful_analyses'] = successful_count
            self.logger.info(f"{logger_prefix} ✓ 所有分析完成（成功 {successful_count}/{len(self.agents)}）")
            
            # ========== 5. 统计结果 ==========
            self.logger.info(f"{logger_prefix} 步骤4: 统计分析结果...")
            
            # 统计知识点总数
            total_knowledge_points = sum(
                analysis.get('knowledge_points_count', 0) 
                for analysis in analyses.values()
            )
            
            self.logger.info(f"{logger_prefix}   参与智能体: {len(self.agents)}")
            self.logger.info(f"{logger_prefix}   成功分析: {successful_count}/{len(self.agents)}")
            self.logger.info(f"{logger_prefix}   知识点总数: {total_knowledge_points}")
            
            # ========== 6. 标记成功 ==========
            result['success'] = True
            self.logger.info(f"{logger_prefix} ==================== 第一轮全体讨论阶段完成 ====================")
            
        except Exception as e:
            error_msg = f"第一轮全体讨论阶段失败: {str(e)}"
            self.logger.error(f"{logger_prefix} ❌ {error_msg}", exc_info=True)
            result['error_message'] = error_msg
            result['success'] = False
        
        return result

    def round_2_team_formation_decide_team_preference(self, task: str, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        第二轮组队阶段 - 每个智能体发表组队意愿
        
        执行流程：
        1. 创建全员团队
        2. 逐个智能体调用 decide_team_preference 方法
        3. 记录组队意愿到消息系统
        4. 统计表态情况
        5. 返回结构化结果
        
        Args:
            task: 任务描述
            analyses: 第一轮讨论的分析结果字典 {agent_id: analysis_result}
            
        Returns:
            Dict[str, Any]: 组队意愿结果字典，包含以下字段：
            {
                'success': bool,  # 是否成功完成
                'preferences': Dict[str, Dict],  # 组队意愿 {agent_id: preference_result}
                'participant_count': int,  # 参与表态的智能体数量
                'successful_count': int,  # 成功表态的智能体数量
                'team_id': str,  # 团队ID
                'timestamp': str,  # 时间戳
                'error_message': str  # 错误信息（如果失败）
            }
        """
        function_name = inspect.currentframe().f_code.co_name
        logger_prefix = f"【{function_name}】"
        
        self.logger.info(f"{logger_prefix} ==================== 组队意愿阶段开始 ====================")
        self.logger.info(f"{logger_prefix} 任务长度: {len(task)} 字符")
        self.logger.info(f"{logger_prefix} 分析结果数量: {len(analyses)} 个")
        
        # 初始化返回结果
        result = {
            'success': False,
            'preferences': {},
            'participant_count': 0,
            'successful_count': 0,
            'team_id': '',
            'timestamp': datetime.now().isoformat(),
            'error_message': ''
        }
        
        try:
            # ========== 1. 创建全员团队 ==========
            self.logger.info(f"{logger_prefix} 步骤1: 创建全员团队...")
            all_agent_ids = [agent.agent_id for agent in self.agents]
            team_id = Team.generate_team_id(all_agent_ids)
            result['team_id'] = team_id
            result['participant_count'] = len(self.agents)
            
            if not self.team_manager.get_team(self.task_id, team_id):
                self.team_manager.create_team(
                    self.task_id,
                    all_agent_ids,
                    RoundStage.ROUND_2_TEAM_FORMATION
                )
                self.logger.debug(f"{logger_prefix}   新建组队团队")
            else:
                self.logger.debug(f"{logger_prefix}   使用已有团队")
            
            self.logger.info(f"{logger_prefix} ✓ 团队ID: {team_id}")
            self.logger.info(f"{logger_prefix}   参与智能体数: {len(self.agents)}")
            self.logger.debug(f"{logger_prefix}   智能体列表: {all_agent_ids}")
            
            # ========== 2. 逐个智能体表态组队意愿 ==========
            self.logger.info(f"{logger_prefix} 步骤2: 收集组队意愿（共 {len(self.agents)} 位智能体）...")
            preferences = {}
            successful_count = 0
            
            for idx, agent in enumerate(self.agents, 1):
                self.logger.debug(f"{logger_prefix}   [{idx}/{len(self.agents)}] {agent.agent_id} 开始表态...")
                
                try:
                    # 设置任务上下文
                    agent.current_task_id = self.task_id
                    
                    # 调用智能体的 decide_team_preference 方法
                    preference_result = agent.decide_team_preference(
                        self.agents, 
                        task, 
                        analyses
                    )
                    
                    # 验证返回值格式
                    if not isinstance(preference_result, dict):
                        self.logger.warning(
                            f"{logger_prefix}   ⚠️ {agent.agent_id} 返回值类型错误: "
                            f"期望 dict，实际 {type(preference_result)}"
                        )
                        # 包装为字典
                        preference_result = {
                            'raw_content': str(preference_result),
                            'agent_id': agent.agent_id
                        }
                    
                    preferences[agent.agent_id] = preference_result
                    successful_count += 1
                    
                    self.logger.debug(f"{logger_prefix}   ✓ {agent.agent_id} 表态完成")
                    
                    # 提取关键信息（如果存在）
                    if 'preferred_teammates' in preference_result:
                        teammates = preference_result['preferred_teammates']
                        self.logger.debug(f"{logger_prefix}     - 期望队友: {teammates}")
                    
                    if 'team_size_preference' in preference_result:
                        team_size = preference_result['team_size_preference']
                        self.logger.debug(f"{logger_prefix}     - 团队规模偏好: {team_size}")
                    
                    # ========== 3. 记录组队意愿到消息系统 ==========
                    # 构建表态摘要（用于消息记录）
                    preference_summary = self._format_preference_summary(
                        agent.agent_id, 
                        preference_result
                    )
                    
                    agent.speak(
                        task_id=self.task_id,
                        team_id=team_id,
                        content=preference_summary,
                        message_type=MessageType.TEAM_FORMATION,
                        round_stage=RoundStage.ROUND_2_TEAM_FORMATION
                    )
                    
                except Exception as exc:
                    error_msg = f"{agent.agent_id} 组队意愿表态失败: {str(exc)}"
                    self.logger.error(f"{logger_prefix}   ❌ {error_msg}", exc_info=True)
                    # 记录失败信息，但继续处理其他智能体
                    preferences[agent.agent_id] = {
                        'success': False,
                        'error': str(exc),
                        'agent_id': agent.agent_id
                    }
            
            result['preferences'] = preferences
            result['successful_count'] = successful_count
            self.logger.info(f"{logger_prefix} ✓ 所有表态完成（成功 {successful_count}/{len(self.agents)}）")
            
            # ========== 4. 统计表态情况 ==========
            self.logger.info(f"{logger_prefix} 步骤3: 统计表态情况...")
            self.logger.info(f"{logger_prefix}   参与智能体: {result['participant_count']}")
            self.logger.info(f"{logger_prefix}   成功表态: {successful_count}/{len(self.agents)}")
            
            if successful_count < len(self.agents):
                failed_count = len(self.agents) - successful_count
                self.logger.warning(f"{logger_prefix}   ⚠️ 有 {failed_count} 位智能体表态失败")
            
            # ========== 5. 标记成功 ==========
            result['success'] = True
            self.logger.info(f"{logger_prefix} ==================== 组队意愿阶段完成 ====================")
            
        except Exception as e:
            error_msg = f"组队意愿阶段失败: {str(e)}"
            self.logger.error(f"{logger_prefix} ❌ {error_msg}", exc_info=True)
            result['error_message'] = error_msg
            result['success'] = False
        
        return result
    
    def _format_preference_summary(self, agent_id: str, preference_result: Dict[str, Any]) -> str:
        """
        格式化组队意愿摘要（用于消息记录）
        
        Args:
            agent_id: 智能体ID
            preference_result: decide_team_preference 返回的结果
            
        Returns:
            格式化的组队意愿摘要字符串
        """
        summary_parts = [f"【{agent_id} 的组队意愿】"]
        
        # 提取关键信息
        if 'preferred_teammates' in preference_result:
            teammates = preference_result['preferred_teammates']
            if isinstance(teammates, list):
                summary_parts.append(f"期望队友: {', '.join(teammates)}")
            else:
                summary_parts.append(f"期望队友: {teammates}")
        
        if 'team_size_preference' in preference_result:
            team_size = preference_result['team_size_preference']
            summary_parts.append(f"团队规模偏好: {team_size}")
        
        if 'reasoning' in preference_result:
            reasoning = preference_result['reasoning']
            summary_parts.append(f"理由: {reasoning}")
        
        # 如果没有提取到关键信息，使用原始内容
        if len(summary_parts) == 1:
            summary_parts.append(str(preference_result))
        
        return "\n".join(summary_parts)
    
    def _transition_from_round_2_team_formation_2_round_3_team_discussion(
        self, 
        task: str, 
        agents_team_preference: Dict[str, Any], 
        n_teams: int = 2
    ) -> Dict[str, Any]:
        """
        第二阶段到第三阶段的过渡，基于组队意愿使用聚类算法形成团队。
        
        Args:
            task: 任务描述
            agents_team_preference: 每个智能体的组队意愿字典
                格式: {agent_id: preference_data}
                其中preference_data包含candidates列表，每个candidate有preference_score
            n_teams: 要分成的团队数量，默认为2
            
        Returns:
            Dict[str, Any]: 包含以下字段的字典
                - success: bool，是否成功分组
                - teams: Dict[str, List[str]]，团队ID到成员列表的映射
                - team_count: int，实际形成的团队数量
                - clustering_method: str，使用的聚类方法
                - clustering_summary: str，聚类过程摘要
                - affinity_matrix_summary: str，亲和度矩阵摘要
                - timestamp: str，操作时间戳
                - error_message: str，错误信息（如果有）
        """
        self.logger.info("【_transition_from_round_2_team_formation_2_round_3_team_discussion】开始组队分组")
        self.logger.info(f"  参与智能体数: {len(agents_team_preference)}")
        self.logger.info(f"  目标团队数: {n_teams}")
        
        try:
            # ========== 步骤1: 构建亲和度矩阵 ==========
            self.logger.debug("【步骤1】构建智能体间的亲和度矩阵...")
            
            agent_ids = list(agents_team_preference.keys())
            n_agents = len(agent_ids)
            
            # 创建亲和度矩阵（对称矩阵）
            affinity_matrix = {}
            for agent_id in agent_ids:
                affinity_matrix[agent_id] = {}
                
            # 填充亲和度矩阵
            for agent_id, preference_data in agents_team_preference.items():
                candidates = preference_data.get('candidates', [])
                
                for candidate in candidates:
                    target_id = candidate.get('agent_id')
                    preference_score = candidate.get('preference_score', 0)
                    
                    if target_id in agent_ids:
                        # 存储A对B的偏好分数
                        affinity_matrix[agent_id][target_id] = preference_score
                        
                        # 如果B对A还没有分数，先设为0（后续会被B的真实评分覆盖）
                        if agent_id not in affinity_matrix[target_id]:
                            affinity_matrix[target_id][agent_id] = 0
            
            # 计算双向平均亲和度（对称化）
            symmetric_affinity = {}
            for i, agent1 in enumerate(agent_ids):
                symmetric_affinity[agent1] = {}
                for j, agent2 in enumerate(agent_ids):
                    if i == j:
                        symmetric_affinity[agent1][agent2] = 0  # 自己对自己的亲和度为0
                    else:
                        # 双向平均
                        score1 = affinity_matrix[agent1].get(agent2, 0)
                        score2 = affinity_matrix[agent2].get(agent1, 0)
                        avg_score = (score1 + score2) / 2.0
                        symmetric_affinity[agent1][agent2] = avg_score
            
            # 生成亲和度矩阵摘要
            affinity_summary = self._format_affinity_matrix_summary(symmetric_affinity, agent_ids)
            self.logger.debug(f"亲和度矩阵构建完成:\n{affinity_summary}")
            
            # ========== 步骤2: 执行贪心聚类算法 ==========
            self.logger.debug(f"【步骤2】执行贪心聚类算法，目标团队数={n_teams}...")
            
            teams_list = self.clustering(symmetric_affinity, agent_ids, n_teams)
            
            self.logger.debug(f"聚类完成，形成 {len(teams_list)} 个团队")
            for idx, team in enumerate(teams_list):
                self.logger.debug(f"  团队{idx+1}: {team}")
            
            # ========== 步骤3: 创建团队记录并生成返回结果 ==========
            self.logger.debug("【步骤3】创建团队记录并生成结构化返回值...")
            
            teams = {}
            for team_members in teams_list:
                if not team_members:  # 跳过空团队
                    continue
                    
                # 生成团队ID
                team_id = Team.generate_team_id(team_members)
                teams[team_id] = team_members
                
                # 创建团队记录
                self.team_manager.create_team(
                    self.task_id,
                    team_members,
                    RoundStage.ROUND_3_TEAM_DISCUSSION
                )
                
                self.logger.info(f"  ✓ 团队创建: {team_id} | 成员: {team_members}")
            
            # 生成聚类摘要
            clustering_summary = self._format_clustering_summary(teams_list, symmetric_affinity)
            
            # 构建返回结果
            result = {
                'success': True,
                'teams': teams,
                'team_count': len(teams),
                'clustering_method': 'greedy_affinity_clustering',
                'clustering_summary': clustering_summary,
                'affinity_matrix_summary': affinity_summary,
                'timestamp': datetime.now().isoformat(),
                'error_message': ''
            }
            
            self.logger.info(f"✅ 组队分组完成: 共形成 {len(teams)} 个团队，{sum(len(m) for m in teams.values())} 名智能体成功组队")
            
            return result
            
        except Exception as e:
            error_msg = f"组队分组过程中发生错误: {e}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'teams': {},
                'team_count': 0,
                'clustering_method': 'greedy_affinity_clustering',
                'clustering_summary': '',
                'affinity_matrix_summary': '',
                'timestamp': datetime.now().isoformat(),
                'error_message': error_msg
            }
    
    def clustering(
        self, 
        affinity_matrix: Dict[str, Dict[str, float]], 
        agent_ids: List[str], 
        n_teams: int
    ) -> List[List[str]]:
        """
        聚类分组。
        
        """
        n_agents = len(agent_ids)
        
        if n_teams >= n_agents:
            return [[agent_id] for agent_id in agent_ids]
        
        if n_teams <= 0:
            n_teams = 1
        
        agent_pairs = []
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids):
                if i < j:  
                    affinity = affinity_matrix[agent1][agent2]
                    agent_pairs.append((affinity, agent1, agent2))
        
        agent_pairs.sort(reverse=True, key=lambda x: x[0])
        
        agent_to_team = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        teams = [[agent_id] for agent_id in agent_ids]
        
        current_team_count = len(teams)
        
        for affinity, agent1, agent2 in agent_pairs:
            if current_team_count <= n_teams:
                break  
            
            team1_idx = agent_to_team[agent1]
            team2_idx = agent_to_team[agent2]
            
            if team1_idx == team2_idx:
                continue  
            
            for agent_id in teams[team2_idx]:
                teams[team1_idx].append(agent_id)
                agent_to_team[agent_id] = team1_idx
            
            teams[team2_idx] = []  
            current_team_count -= 1
        final_teams = [team for team in teams if team]
        
        return final_teams
    
    def _format_affinity_matrix_summary(
        self, 
        affinity_matrix: Dict[str, Dict[str, float]], 
        agent_ids: List[str]
    ) -> str:
        """
        格式化亲和度矩阵摘要。
        
        Args:
            affinity_matrix: 亲和度矩阵
            agent_ids: 智能体ID列表
            
        Returns:
            str: 格式化的摘要文本
        """
        summary_lines = ["亲和度矩阵（双向平均）:"]
        summary_lines.append("=" * 60)
        
        # 表头
        header = "        "
        for agent_id in agent_ids:
            header += f"{agent_id:>12}"
        summary_lines.append(header)
        summary_lines.append("-" * 60)
        
        # 矩阵内容
        for agent1 in agent_ids:
            row = f"{agent1:>8}"
            for agent2 in agent_ids:
                score = affinity_matrix[agent1][agent2]
                row += f"{score:>12.2f}"
            summary_lines.append(row)
        
        summary_lines.append("=" * 60)
        
        # 最高亲和度对
        max_affinity = 0
        max_pair = None
        for agent1 in agent_ids:
            for agent2 in agent_ids:
                if agent1 < agent2:  # 避免重复
                    affinity = affinity_matrix[agent1][agent2]
                    if affinity > max_affinity:
                        max_affinity = affinity
                        max_pair = (agent1, agent2)
        
        if max_pair:
            summary_lines.append(f"最高亲和度对: {max_pair[0]} ↔ {max_pair[1]} (分数: {max_affinity:.2f})")
        
        return "\n".join(summary_lines)
    
    def _format_clustering_summary(
        self, 
        teams: List[List[str]], 
        affinity_matrix: Dict[str, Dict[str, float]]
    ) -> str:
        """
        格式化聚类结果摘要。
        
        Args:
            teams: 团队列表
            affinity_matrix: 亲和度矩阵
            
        Returns:
            str: 格式化的摘要文本
        """
        summary_lines = ["聚类结果摘要:"]
        summary_lines.append("=" * 60)
        
        for idx, team in enumerate(teams):
            if not team:
                continue
                
            summary_lines.append(f"\n团队 {idx + 1}: ({len(team)} 名成员)")
            summary_lines.append(f"  成员: {', '.join(team)}")
            
            # 计算团队内部平均亲和度
            if len(team) > 1:
                total_affinity = 0
                pair_count = 0
                for i, agent1 in enumerate(team):
                    for j, agent2 in enumerate(team):
                        if i < j:
                            total_affinity += affinity_matrix[agent1][agent2]
                            pair_count += 1
                
                avg_affinity = total_affinity / pair_count if pair_count > 0 else 0
                summary_lines.append(f"  团队内平均亲和度: {avg_affinity:.2f}")
            else:
                summary_lines.append(f"  团队内平均亲和度: N/A (单人团队)")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)

    def round_3_team_discussion(self, task: str, teams: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        第三阶段：小组讨论阶段
        
        每个团队内部讨论形成方案，并基于讨论过程中收集的代表意愿度选出团队代表
        
        执行流程：
        1. 遍历每个团队，组织内部讨论
        2. 讨论完成后，从representative_intentions中选出意愿度最高的成员作为代表
        3. 返回结构化的讨论结果
        
        Args:
            task: 任务描述
            teams: 形成的团队字典 {team_id: [agent_id列表]}
        
        Returns:
            Dict[str, Any]: 讨论结果字典，包含以下字段：
            {
                'success': bool,  # 是否成功完成
                'team_results': Dict[str, Dict],  # 各团队的讨论结果
                'representatives': List[str],  # 各团队代表的agent_id列表
                'team_count': int,  # 团队数量
                'successful_teams': int,  # 成功完成讨论的团队数
                'timestamp': str,  # 完成时间戳
                'error_message': str  # 错误信息（如果有）
            }
        """
        self.logger.info("="*80)
        self.logger.info("🌟 ROUND 3: 小组讨论阶段")
        self.logger.info("="*80)
        self.logger.debug(f"  团队数量: {len(teams)}")
        self.logger.debug(f"  团队信息: {teams}")
        
        try:
            # ========== 初始化结果收集器 ==========
            team_results: Dict[str, Dict[str, Any]] = {}
            representatives: List[str] = []
            successful_teams = 0
            
            # ========== 遍历每个团队进行讨论 ==========
            for team_id, members in teams.items():
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"👥 团队 {team_id} 内部讨论")
                self.logger.info(f"{'='*80}")
                self.logger.debug(f"  成员: {members}")
                
                # ===== 步骤1: 团队内部讨论 =====
                self.logger.debug("【步骤1】执行团队内部讨论...")
                discussion_result = self._internal_teams_discussion(
                    task=task,
                    members=members,
                    stage=RoundStage.ROUND_3_TEAM_DISCUSSION,
                    num_listeners=6
                )
                
                # 检查讨论是否成功
                if not discussion_result['success']:
                    self.logger.error(f"❌ 团队 {team_id} 讨论失败: {discussion_result['error_message']}")
                    team_results[team_id] = {
                        'success': False,
                        'error_message': discussion_result['error_message'],
                        'representative': None
                    }
                    continue
                
                successful_teams += 1
                self.logger.info(f"✓ 团队 {team_id} 讨论完成")
                self.logger.info(f"  轮次: {discussion_result['total_rounds']}/{discussion_result['max_rounds']}")
                self.logger.info(f"  共识: {discussion_result['consensus_achieved']}")
                
                # ===== 步骤2: 从代表意愿度中选出代表 =====
                self.logger.debug("【步骤2】基于代表意愿度选举团队代表...")
                
                representative_intentions = discussion_result.get('representative_intentions', {})
                
                if not representative_intentions:
                    # 如果没有收集到代表意愿度，默认选第一个成员
                    representative_id = members[0] if members else None
                    self.logger.warning(f"⚠️ 未收集到代表意愿度信息，默认选择第一个成员: {representative_id}")
                else:
                    # 选出representative_score最高的成员
                    representative_id = max(
                        representative_intentions.keys(),
                        key=lambda agent_id: representative_intentions[agent_id].get('representative_score', 0)
                    )
                    
                    rep_info = representative_intentions[representative_id]
                    self.logger.info(f"✅ 团队 {team_id} 代表选举完成")
                    self.logger.info(f"  代表: {representative_id}")
                    self.logger.info(f"  意愿度: {rep_info.get('representative_degree', '未知')} ({rep_info.get('representative_score', 0)}分)")
                    self.logger.info(f"  理由: {rep_info.get('representative_reason', '无')[:100]}")
                    
                    # 输出所有成员的代表意愿度（用于调试）
                    self.logger.debug("  所有成员的代表意愿度:")
                    for agent_id, intention in representative_intentions.items():
                        self.logger.debug(f"    - {agent_id}: {intention.get('representative_score', 0)}分 ({intention.get('representative_degree', '未知')})")
                
                # 记录代表
                representatives.append(representative_id)
                
                # 设置智能体的代表角色
                representative_agent = self._get_agent_by_id(representative_id)
                if representative_agent:
                    representative_agent.representative_of_team = team_id
                    representative_agent.representative_of_members = members
                    self.logger.debug(f"  已设置 {representative_id} 为团队 {team_id} 的代表")
                
                # 保存团队结果
                team_results[team_id] = {
                    'success': True,
                    'discussion_result': discussion_result,
                    'representative': representative_id,
                    'representative_intentions': representative_intentions
                }
            
            # ========== 构建最终返回结果 ==========
            result = {
                'success': True,
                'team_results': team_results,
                'representatives': representatives,
                'team_count': len(teams),
                'successful_teams': successful_teams,
                'timestamp': datetime.now().isoformat(),
                'error_message': ''
            }
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✅ 第三轮讨论完成")
            self.logger.info(f"  总团队数: {len(teams)}")
            self.logger.info(f"  成功团队数: {successful_teams}")
            self.logger.info(f"  选出代表: {representatives}")
            self.logger.info(f"{'='*80}")
            
            return result
            
        except Exception as e:
            error_msg = f"第三轮讨论时发生错误: {e}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            return {
                'success': False,
                'team_results': team_results if 'team_results' in locals() else {},
                'representatives': representatives if 'representatives' in locals() else [],
                'team_count': len(teams),
                'successful_teams': successful_teams if 'successful_teams' in locals() else 0,
                'timestamp': datetime.now().isoformat(),
                'error_message': error_msg
            }

    def _internal_teams_discussion(
        self, 
        task: str, 
        members: List[str], 
        stage: RoundStage, 
        max_rounds: int = 0, 
        num_listeners: int = 6
    ) -> Dict[str, Any]:
        """
        团队内部讨论 - 智能体在团队内进行多轮讨论直到达成共识
        
        执行流程：
        1. 初始化团队和发言队列
        2. 循环执行：获取发言者 → 生成内容 → 记录发言 → 态度评估 → 共识判断
        3. 达成共识或达到最大轮次后结束
        4. 返回结构化讨论结果
        
        Args:
            task: 任务描述
            members: 团队成员ID列表
            stage: 讨论阶段（ROUND_3_TEAM_DISCUSSION 或 ROUND_4_FINAL_DISCUSSION）
            max_rounds: 最大讨论轮次（0表示使用系统配置）
            num_listeners: 每轮随机抽样评估共识的成员数量
            
        Returns:
            Dict[str, Any]: 讨论结果字典，包含以下字段：
            {
                'success': bool,  # 讨论是否成功完成
                'team_id': str,  # 团队ID
                'discussion_records': List[Dict],  # 讨论记录列表
                'consensus_achieved': bool,  # 是否达成共识
                'consensus_round': int,  # 达成共识的轮次（如果有）
                'total_rounds': int,  # 实际讨论轮次
                'max_rounds': int,  # 最大允许轮次
                'consensus_status': Dict[str, bool],  # 各成员的共识状态
                'participant_count': int,  # 参与成员数量
                'termination_reason': str,  # 结束原因
                'timestamp': str,  # 讨论结束时间戳
                'error_message': str  # 错误信息（如果有）
            }
        """
        self.logger.info("【_internal_teams_discussion】开始团队内部讨论")
        self.logger.info(f"  成员数: {len(members)}")
        self.logger.info(f"  讨论阶段: {stage}")
        
        try:
            # ========== 步骤1: 初始化团队和智能体 ==========
            self.logger.debug("【步骤1】初始化团队和智能体...")
            
            team_agents = [self._get_agent_by_id(agent_id) for agent_id in members]
            team_agents = [agent for agent in team_agents if agent is not None]
            
            if not team_agents:
                error_msg = "无有效的团队成员，无法进行小组讨论"
                self.logger.error(f"❌ {error_msg}")
                return {
                    'success': False,
                    'team_id': '',
                    'discussion_records': [],
                    'consensus_achieved': False,
                    'consensus_round': 0,
                    'total_rounds': 0,
                    'max_rounds': 0,
                    'consensus_status': {},
                    'participant_count': 0,
                    'termination_reason': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'error_message': error_msg
                }
            
            team_id = Team.generate_team_id(members)
            team = self.team_manager.get_team(self.task_id, team_id)
            if not team:
                team = self.team_manager.create_team(
                    self.task_id,
                    members,
                    stage
                )
            
            self.logger.info(f"✓ 团队初始化成功: {team_id}")
            self.logger.debug(f"  参与成员: {[a.agent_id for a in team_agents]}")
            
            # ========== 步骤2: 初始化讨论状态 ==========
            self.logger.debug("【步骤2】初始化讨论状态...")
            
            discussion_records: List[Dict[str, Any]] = []
            consensus_status: Dict[str, bool] = {agent.agent_id: False for agent in team_agents}
            representative_intentions: Dict[str, Dict[str, Any]] = {}  # 收集各成员的代表意愿度
            member_map = {agent.agent_id: agent for agent in team_agents}
            
            # 计算最大轮次
            max_rounds = max_rounds if max_rounds > 0 else self.system_config.max_rounds
            self.logger.info(f"  最大讨论轮次: {max_rounds}")
            
            # 初始化发言队列
            self.speaking_queue.clear_queue()
            for agent in team_agents:
                agent.speaking_intention.intention_score = agent.speaking_intention.threshold
                self.speaking_queue.update_intention(agent.agent_id, agent.speaking_intention)
                agent.current_task_id = self.task_id
            
            self.logger.info(f"✓ 讨论状态初始化完成")
            
            # ========== 步骤3: 多轮讨论循环 ==========
            self.logger.debug("【步骤3】开始多轮讨论...")
            
            round_index = 0
            speakers_this_round: Set[str] = set()
            spoke_in_round = False
            empty_round_count = 0
            max_empty_rounds = 5
            consensus_achieved = False
            consensus_round = 0
            termination_reason = ""
            
            while round_index < max_rounds:
                # 所有成员都已经在本轮发言过，重置到下一轮
                if len(speakers_this_round) == len(member_map):
                    self.logger.debug(f"本轮所有成员已发言，开始新一轮")
                    speakers_this_round.clear()
                    spoke_in_round = False
                
                # 获取下一个发言者
                next_speaker_id = self.speaking_queue.get_next_speaker()
                self.logger.debug(f"【start speaker {next_speaker_id}】 ====================================================================================================================================================================================================")
                self.logger.debug(f"队列中下一个发言者: {next_speaker_id}")
                
                # 处理无效发言者
                if next_speaker_id not in member_map:
                    empty_round_count += 1
                    self.logger.debug(f"获取到非本团队成员，空轮计数: {empty_round_count}/{max_empty_rounds}")
                    
                    if empty_round_count >= max_empty_rounds:
                        termination_reason = f"连续 {empty_round_count} 次无法获取有效发言者"
                        self.logger.warning(f"⚠️ {termination_reason}，退出讨论")
                        break
                    
                    # 唤醒成员
                    if not spoke_in_round:
                        target_agents = team_agents
                    else:
                        target_agents = [member_map[aid] for aid, ok in consensus_status.items() if not ok]
                    
                    for target_agent in target_agents:
                        target_agent.speaking_intention.intention_score = target_agent.speaking_intention.threshold
                        self.speaking_queue.update_intention(target_agent.agent_id, target_agent.speaking_intention)
                    
                    continue
                
                # 处理已发言成员
                if next_speaker_id in speakers_this_round:
                    self.logger.debug(f"{next_speaker_id} 已在本轮发言，跳过")
                    
                    # 唤醒未发言成员
                    unspoken_agents = [
                        member_map[aid] for aid in member_map.keys() 
                        if aid not in speakers_this_round
                    ]
                    
                    for unspoken_agent in unspoken_agents:
                        if not consensus_status.get(unspoken_agent.agent_id, False):
                            unspoken_agent.speaking_intention.intention_score = unspoken_agent.speaking_intention.threshold
                            self.speaking_queue.update_intention(unspoken_agent.agent_id, unspoken_agent.speaking_intention)
                    
                    continue
                
                # 成功获取有效发言者
                empty_round_count = 0
                round_index += 1
                self.logger.info(f"  📢 第 {round_index}/{max_rounds} 轮发言")
                
                agent = member_map[next_speaker_id]
                speakers_this_round.add(agent.agent_id)
                spoke_in_round = True
                
                # 生成讨论内容
                try:
                    self.logger.debug(f"    {agent.agent_id} 正在生成发言...")
                    generate_result = agent.generate(task, round_index=round_index, max_rounds=max_rounds)
                    
                    # 适配 generate 的返回值（可能是字典或字符串）
                    if isinstance(generate_result, dict):
                        content = generate_result.get('main_solution', '') or generate_result.get('content', '')
                    else:
                        content = str(generate_result)
                    
                    self.logger.info(f"    ✓ {agent.agent_id} 发言成功（{len(content)} 字符）")
                    
                except Exception as exc:
                    error_content = f"【错误】生成讨论内容失败: {exc}"
                    self.logger.error(f"    ❌ {agent.agent_id} 生成讨论内容失败: {exc}", exc_info=True)
                    content = error_content
                
                # 记录发言
                discussion_records.append({
                    "round": round_index,
                    "agent_id": agent.agent_id,
                    "content": content
                })
                
                # 发送消息
                message = agent.speak(
                    task_id=self.task_id,
                    team_id=team.team_id,
                    content=content,
                    message_type=MessageType.DISCUSSION,
                    round_stage=stage
                )
                
                # 性能优化：如果已是最后一轮，直接结束，无需进行态度评估和共识判断
                if round_index >= max_rounds:
                    termination_reason = f"达到最大讨论轮次 ({max_rounds})"
                    self.logger.info(f"  ⏱️ {termination_reason}，跳过态度评估直接结束")
                    break
                
                # 态度评估和共识判断
                n = num_listeners
                self.logger.debug(f"    随机抽样 {n} 位成员评估共识...")
                selected_listeners = random.sample(team_agents, min(n, len(team_agents)))
                selected_consensus_achieved = True
                
                for listener in selected_listeners:
                    try:
                        # 调用 attitude 函数（返回字典）
                        attitude_result = listener.attitude(task, message, round_index, max_rounds)
                        
                        if not attitude_result.get('success', False):
                            self.logger.warning(f"      ⚠️ {listener.agent_id} 态度分析失败")
                            selected_consensus_achieved = False
                            continue
                        
                        # 提取发言意愿信息
                        speaking_intention = attitude_result.get('speaking_intention', {})
                        intention_score = speaking_intention.get('intention_score', 6.0)
                        intention_reason = speaking_intention.get('intention_reason', '无具体理由')
                        
                        # 更新发言意愿
                        listener.update_speaking_intention(intention_score, intention_reason)
                        
                        # 提取代表意愿度信息（新增）
                        representative_intention = attitude_result.get('representative_intention', {})
                        if representative_intention:
                            # 保存或更新该成员的代表意愿度（保留最新的）
                            representative_intentions[listener.agent_id] = {
                                'desire_to_represent': representative_intention.get('desire_to_represent', False),
                                'representative_degree': representative_intention.get('representative_degree', '中等'),
                                'representative_score': representative_intention.get('representative_score', 6.0),
                                'representative_reason': representative_intention.get('representative_reason', '无理由'),
                                'round': round_index  # 记录是在哪一轮获取的
                            }
                            self.logger.debug(f"      {listener.agent_id}: 代表意愿度={representative_intention.get('representative_score', 0)}")
                        
                        # 提取讨论评估信息（判断是否达成共识）
                        discussion_eval = attitude_result.get('discussion_evaluation', {})
                        completeness = discussion_eval.get('completeness_degree', '部分完善')
                        
                        # 根据完善度判断共识
                        # 简化判断：如果完善度为"非常完善"或"基本完善"，则认为达成共识
                        is_consensus = completeness in ['非常完善', '基本完善']
                        
                        consensus_status[listener.agent_id] = is_consensus
                        
                        if not is_consensus:
                            selected_consensus_achieved = False
                        
                        self.logger.debug(f"      {listener.agent_id}: 完善度={completeness}, 共识={is_consensus}")
                        
                    except Exception as exc:
                        self.logger.warning(f"      ⚠️ {listener.agent_id} 态度评估失败: {exc}")
                        selected_consensus_achieved = False
                
                # 判断是否达成共识
                if selected_consensus_achieved:
                    consensus_achieved = True
                    consensus_round = round_index
                    termination_reason = f"在第 {round_index} 轮达成共识（基于随机抽样）"
                    self.logger.info(f"  ✅ {termination_reason}")
                    break
                
                # 唤醒未达成共识的成员
                target_agents = [
                    listener for listener in selected_listeners 
                    if not consensus_status.get(listener.agent_id, False)
                ]
                
                for target_agent in target_agents:
                    target_agent.speaking_intention.intention_score = target_agent.speaking_intention.threshold
                    self.speaking_queue.update_intention(target_agent.agent_id, target_agent.speaking_intention)
                self.logger.debug(f"【end speaker {next_speaker_id}】 ====================================================================================================================================================================================================")

            # ========== 步骤4: 清理和总结 ==========
            self.logger.debug("【步骤4】清理队列并生成讨论总结...")
            
            # 清理队列
            self.speaking_queue.clear_queue()
            
            # 确定结束原因
            if not termination_reason:
                if round_index >= max_rounds:
                    termination_reason = f"达到最大讨论轮次 ({max_rounds})"
                else:
                    termination_reason = "未知原因"
            
            # 构建返回结果
            result = {
                'success': True,
                'team_id': team_id,
                'discussion_records': discussion_records,
                'consensus_achieved': consensus_achieved,
                'consensus_round': consensus_round,
                'total_rounds': round_index,
                'max_rounds': max_rounds,
                'consensus_status': consensus_status,
                'representative_intentions': representative_intentions,  # 新增：代表意愿度信息
                'participant_count': len(team_agents),
                'termination_reason': termination_reason,
                'timestamp': datetime.now().isoformat(),
                'error_message': ''
            }
            
            self.logger.info(f"✅ 团队内部讨论完成")
            self.logger.info(f"  实际轮次: {round_index}/{max_rounds}")
            self.logger.info(f"  达成共识: {consensus_achieved}")
            self.logger.info(f"  收集到 {len(representative_intentions)} 个成员的代表意愿度")
            self.logger.info(f"  结束原因: {termination_reason}")
            
            return result
            
        except Exception as e:
            error_msg = f"团队内部讨论时发生错误: {e}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            return {
                'success': False,
                'team_id': team_id if 'team_id' in locals() else '',
                'discussion_records': discussion_records if 'discussion_records' in locals() else [],
                'consensus_achieved': False,
                'consensus_round': 0,
                'total_rounds': round_index if 'round_index' in locals() else 0,
                'max_rounds': max_rounds if 'max_rounds' in locals() else 0,
                'consensus_status': consensus_status if 'consensus_status' in locals() else {},
                'representative_intentions': representative_intentions if 'representative_intentions' in locals() else {},
                'participant_count': len(members),
                'termination_reason': error_msg,
                'timestamp': datetime.now().isoformat(),
                'error_message': error_msg
            }

    def round_4_final_discussion(self, task: str, representation_agents: List[str]) -> Dict[str, Any]:
        """
        第四轮: 代表讨论。各团队代表形成团队，然后讨论形成最终方案。
        
        执行流程：
        1. 创建代表团队
        2. 组织代表内部讨论（调用 _internal_teams_discussion）
        3. 从 representative_intentions 中选出意愿度最高的代表撰写最终答案
        4. 由选出的代表整理最终答案
        
        Args:
            task: 任务描述
            representation_agents: 每个团队中被选为代表的智能体agent_id列表
            
        Returns:
            Dict[str, Any]: 包含最终讨论结果的字典
            {
                'success': bool,  # 讨论是否成功
                'team_id': str,  # 代表团队ID
                'discussion_result': Dict,  # 讨论详细结果
                'final_representative': str,  # 最终选出的撰写代表
                'final_answer': str,  # 最终答案
                'representative_intentions': Dict,  # 代表意愿度信息
                'representative_count': int,  # 参与代表数量
                'timestamp': str,  # 完成时间戳
                'error_message': str  # 错误信息
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{task_name}】"
        
        try:
            self.logger.info(f"🎯 {logger_name} 开始第四轮代表讨论")
            self.logger.info(f"  参与代表数量: {len(representation_agents)}")
            self.logger.debug(f"  代表列表: {representation_agents}")
            
            # ========== 步骤1: 创建代表团队 ==========
            team_id = Team.generate_team_id(representation_agents)
            if not self.team_manager.get_team(self.task_id, team_id):
                self.team_manager.create_team(
                    self.task_id,
                    representation_agents,
                    RoundStage.ROUND_4_FINAL_DISCUSSION
                )
                self.logger.debug(f"✓ 创建代表团队: {team_id}")
            else:
                self.logger.debug(f"✓ 代表团队已存在: {team_id}")
            
            # ========== 步骤2: 组织代表内部讨论 ==========
            self.logger.debug(f"\n{'='*60}")
            self.logger.info(f"👥 步骤1: 代表团队 {team_id} 内部讨论")
            self.logger.debug(f"{'='*60}\n")
            
            max_rounds = self.system_config.max_final_discussion_rounds
            self.logger.debug(f"  最大讨论轮次: {max_rounds}")
            
            discussion_result = self._internal_teams_discussion(
                task=task,
                members=representation_agents,
                stage=RoundStage.ROUND_4_FINAL_DISCUSSION,
                max_rounds=max_rounds,
                num_listeners=2
            )
            
            # 检查讨论是否成功
            if not discussion_result['success']:
                error_msg = f"代表讨论失败: {discussion_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                return {
                    'success': False,
                    'team_id': team_id,
                    'discussion_result': discussion_result,
                    'final_representative': '',
                    'final_answer': '',
                    'representative_intentions': {},
                    'representative_count': len(representation_agents),
                    'timestamp': datetime.now().isoformat(),
                    'error_message': error_msg
                }
            
            discussion_records = discussion_result['discussion_records']
            self.logger.info(f"✓ 代表讨论完成: {discussion_result['total_rounds']}/{max_rounds} 轮")
            self.logger.info(f"  共识达成: {discussion_result['consensus_achieved']}")
            self.logger.debug(f"  讨论记录数: {len(discussion_records)}")
            
            # ========== 步骤3: 从 representative_intentions 中选出撰写代表 ==========
            self.logger.debug(f"\n{'-'*60}")
            self.logger.info(f"🗳️ 步骤2: 选举最终撰写代表")
            self.logger.debug(f"{'-'*60}\n")
            
            representative_intentions = discussion_result.get('representative_intentions', {})
            
            if not representative_intentions:
                # 如果没有收集到代表意愿度，默认选第一个代表
                self.logger.warning(f"⚠️ 未收集到代表意愿度信息，默认选择第一个代表")
                final_representative = representation_agents[0]
            else:
                # 输出所有代表的意愿度（便于调试）
                self.logger.debug(f"收集到 {len(representative_intentions)} 个代表的意愿度:")
                for agent_id, intention in representative_intentions.items():
                    score = intention.get('representative_score', 0)
                    degree = intention.get('representative_degree', '未知')
                    reason = intention.get('representative_reason', '无')
                    self.logger.debug(f"  - {agent_id}: {score}分 ({degree}) - {reason}")
                
                # 选取 representative_score 最高的代表
                final_representative = max(
                    representative_intentions.keys(),
                    key=lambda agent_id: representative_intentions[agent_id].get('representative_score', 0)
                )
                
                selected_score = representative_intentions[final_representative].get('representative_score', 0)
                selected_degree = representative_intentions[final_representative].get('representative_degree', '未知')
                
                self.logger.info(f"✓ 选出最终撰写代表: {final_representative}")
                self.logger.info(f"  代表意愿度: {selected_score}分 ({selected_degree})")
                self.logger.debug(f"  选举理由: {representative_intentions[final_representative].get('representative_reason', '无')}")
            
            # ========== 步骤4: 由选出的代表整理最终答案 ==========
            self.logger.debug(f"\n{'-'*60}")
            self.logger.info(f"📝 步骤3: 由代表 {final_representative} 整理最终答案")
            self.logger.debug(f"{'-'*60}\n")
            
            final_answer = self._summarize_final_answer(task, final_representative)
            
            self.logger.info(f"✓ 最终答案已生成")
            self.logger.debug(f"  答案长度: {len(final_answer)} 字符")
            
            # ========== 构建返回结果 ==========
            result = {
                'success': True,
                'team_id': team_id,
                'discussion_result': discussion_result,
                'final_representative': final_representative,
                'final_answer': final_answer,
                'representative_intentions': representative_intentions,
                'representative_count': len(representation_agents),
                'timestamp': datetime.now().isoformat(),
                'error_message': ''
            }
            
            self.logger.info(f"✅ {logger_name} 第四轮代表讨论完成")
            self.logger.debug(f"  最终代表: {final_representative}")
            self.logger.debug(f"  答案长度: {len(final_answer)} 字符")
            
            return result
            
        except Exception as e:
            error_msg = f"第四轮讨论过程发生异常: {str(e)}"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            import traceback
            self.logger.debug(f"异常堆栈:\n{traceback.format_exc()}")
            
            return {
                'success': False,
                'team_id': team_id if 'team_id' in locals() else '',
                'discussion_result': discussion_result if 'discussion_result' in locals() else {},
                'final_representative': '',
                'final_answer': '',
                'representative_intentions': {},
                'representative_count': len(representation_agents),
                'timestamp': datetime.now().isoformat(),
                'error_message': error_msg
            }
    
    def _summarize_final_answer(self, task: str, representative_agent_id: str) -> str:
        '''
        由代表来整理最终答案。
        Args:
            task: 任务描述
            representative_agent_id: 代表智能体ID
            disscusion: 讨论记录列表。每个Dict包含"round"（轮次）、"agent_id"（发言者ID）、"content"（发言内容）。
        Returns:
            str: 最终答案
        '''
        representative_agent = self._get_agent_by_id(representative_agent_id)
        if not representative_agent:
            raise ValueError(f"代表智能体 {representative_agent_id} 未找到，无法整理最终答案")
        self.logger.info(f"📝 代表 {representative_agent_id} 开始整理最终答案..." )
        task =task +"请基讨论的历史记录，整理出最终的解决方案：\n" 
        is_single = True # TODO 直接由大模型生成答案，不再分段生成。TODO 这里可以配置是否分段生成,放到配置文件中。
        min_words = 4000 # TODO 这里可以配置最少字数,放到配置文件中。
        if is_single:
            collate_result = representative_agent.collate_single(task,min_words=min_words)
        else:
            collate_result = representative_agent.collate(task,min_words=min_words) 
        self.logger.info(f"✅ 代表 {representative_agent_id} 整理最终答案完成。" )
        
        if collate_result.get('success', False):
            final_content = collate_result.get('article_text', '')
        else:
            final_content = f"整理最终答案失败: {collate_result.get('error_message', '未知错误')}"

        representative_agent.speak(
            task_id=self.task_id,
            team_id=Team.generate_team_id([representative_agent_id]),
            content=final_content,
            message_type=MessageType.FINAL_ANSWER,
            round_stage=RoundStage.ROUND_4_FINAL_DISCUSSION
        )
        
        return collate_result
    

