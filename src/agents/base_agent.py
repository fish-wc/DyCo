"""
智能体基类
所有MBTI智能体的基类,提供通用功能
"""
import re
import json
import logging
import inspect
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import inspect

from src.variables.tokens import record_token_usage
from src.utils.helpers import get_tag_content
from src.utils.discussion_summary_manager import DiscussionSummaryManager
from ..models.config import AgentConfig, SystemConfig
from ..models.message import Message, MessageType, RoundStage, SpeakingIntention
from ..communication.message_manager import MessageManager
from ..communication.speaking_queue import SpeakingQueue
from ..utils.llm_client import create_llm_client, call_llm, LLMToolClient
from ..prompts import prompt_loader
from ..tools.system.knowledgemanager import (add_knowledge_tool,# 工具管理器，这里导入工具
                                            add_knowledge_batch_tool,
                                            query_knowledge_tool,
                                            )
from src.tools.system.websearch import web_search_tool


# smolagents 导入 TODO 这部分代码多余，后面考虑重构删除这块的内容。
try:
    # from smolagents import CodeAgent, tool 。这里将CodeAgent改为ToolCallingAgent，发现采用json形式调用工具会更稳定一些。因为采用code形式调用工具会有语法错误的问题，会导致多余的超时重试。
    from smolagents import ToolCallingAgent
    
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    CodeAgent = None
    tool = None

stages =[RoundStage.INITIAL_SOLUTION,
        RoundStage.ROUND_1_DISCUSSION,
        RoundStage.ROUND_3_TEAM_DISCUSSION,
        RoundStage.ROUND_4_FINAL_DISCUSSION]
# ROUND_2_TEAM_FORMATION  ; 不纳入记忆管理。

class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, config: AgentConfig, 
                 message_manager: MessageManager,
                 speaking_queue: SpeakingQueue,
                 logger: Optional[logging.Logger] = None,
                 system_config: SystemConfig = None,
                 knowledge_manager: Optional[Any] = None):
        """
        初始化智能体
        
        Args:
            config: 智能体配置
            message_manager: 消息管理器
            speaking_queue: 发言队列。为了避免智能体之间无规则抢话题冲突。
            logger: 日志记录器
            system_config: 系统配置
            knowledge_manager: 知识库管理器实例
        """
        self.config = config
        self.agent_id = config.agent_id
        self.agent_name = config.agent_name
        self.mbti_type = config.mbti_type
        self.system_config = system_config or SystemConfig()
        
        # 日志记录器（使用传入的logger或获取默认logger）
        self.logger = logger or logging.getLogger(f"agent.{self.agent_id}")
        
        self.logger.info(f"初始化智能体: {self.agent_id} ({self.agent_name})。MBTI类型: {self.mbti_type}。")
        
        # 通信组件 TODO 消息队列可以考虑从agent里面去掉
        self.message_manager = message_manager
        self.speaking_queue = speaking_queue
        self.logger.debug("消息管理器和发言队列已注入")
     
        # 知识库管理器
        self.knowledge_manager = knowledge_manager
        if knowledge_manager:
            self.logger.info(f"知识库管理器已注入")
        else:
            self.logger.warning("未提供知识库管理器，知识管理功能将不可用")
        
        # 讨论摘要管理器（用于收集关键信息）
        self.discussion_summary_manager: Optional[DiscussionSummaryManager] = None
     
        # 发言意愿
        self.speaking_intention = SpeakingIntention(
            agent_id=self.agent_id,
            threshold=config.personality.speaking_threshold
        )
        self.logger.debug(f"发言意愿阈值: {config.personality.speaking_threshold}")
        
        # 发言思路（用于 generate 函数检索知识库）
        self.speaking_plan_context: str = ""
        
        # 当前任务ID
        self.current_task_id: Optional[str] = None
        
        # smolagents 相关
        self._smolagent: Optional[Any] = None  # CodeAgent实例
        self._smolagent_tools: List[Any] = []  # 注册的工具列表
        self._count_system_tools = 0 # 系统工具数量，该数值之后的就是用户自定义工具
        self._init_smolagent_tools()
        self.register_tools()
        
        # LLM客户端初始化
        self._init_llm_client()
        
        # LLMToolClient 初始化 - 用于结构化的大模型管理
        self._init_llm_tool_client()
        
        # Collate专用LLM客户端初始化 - 用于整理最终文章
        self._init_collate_llm_client()
        
        # 加一个团队代表的标识
        self.representative_of_team: Optional[str] = None # 代表哪个团队
        self.representative_of_members: Optional[List[str]] = None # 代表哪些成员
        
        self.logger.info(f"✅ 智能体 {self.agent_id} 初始化完成")
    
    def _init_smolagent_tools(self):
        """初始化smolagent工具 - 子类可以重写此方法添加自定义工具"""
        self.logger.debug("初始化smolagent工具（基类默认实现）...")
        # 基类默认不添加任何工具，子类可以重写此方法
        pass
    
    def register_tools(self):
        """注册工具 - 子类需要重写此方法注册自己的工具"""
        self.logger.debug("注册工具（基类默认实现）...")
        # 基类默认不注册任何工具，子类需要重写
        pass
    
    def _register_smolagent_tool(self, tools: List[Any]):
        """注册smolagent工具
        
        Args:
            tools: 工具列表
        """
        self.logger.debug(f"注册 {len(tools)} 个smolagent工具...")
        # 这是一个简化的实现，实际可能需要更复杂的逻辑
        self._smolagent_tools.extend(tools)
        self.logger.info(f"✓ 成功注册 {len(tools)} 个工具")
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        self.logger.debug("开始初始化LLM客户端...")
        try:
            from ..utils.llm_client import create_llm_client
            self.llm_client = create_llm_client(self.config.model)
            
            self.logger.info(f"✓ LLM客户端初始化成功")
            self.logger.debug(f"  模型: {self.config.model.model_name}")
            self.logger.debug(f"  URL: {self.config.model.model_url}")
        except Exception as e:
            self.logger.error(f"❌ LLM客户端初始化失败: {e}")
            self.llm_client = None
    
    def _init_llm_tool_client(self):
        """初始化 LLMToolClient - 用于结构化的大模型管理"""
        self.logger.debug("开始初始化 LLMToolClient...")
        try:
            self.llm_tool_client = LLMToolClient(
                config=self.config.model,
                use_tools_api=True,
                logger=self.logger
            )
            self.logger.info(f"✓ LLMToolClient 初始化成功")
            self.logger.debug(f"  支持工具调用API")
        except Exception as e:
            self.logger.error(f"❌ LLMToolClient 初始化失败: {e}")
            self.llm_tool_client = None
    
    def _init_collate_llm_client(self):
        """初始化 Collate 专用 LLM 客户端 - 用于整理最终文章"""
        self.logger.debug("开始初始化 Collate 专用 LLM 客户端...")
        try:
            # 如果系统配置中有 collate_model，使用专用配置；否则使用默认模型
            if self.system_config and hasattr(self.system_config, 'collate_model') and self.system_config.collate_model:
                collate_model_config = self.system_config.collate_model
                self.logger.info(f"  使用专用 Collate 模型: {collate_model_config.model_name}")
            else:
                collate_model_config = self.config.model
                self.logger.info(f"  未配置专用 Collate 模型，使用默认模型: {collate_model_config.model_name}")
            
            from ..utils.llm_client import create_llm_client
            self.collate_llm_client = create_llm_client(collate_model_config)
            
            self.logger.info(f"✓ Collate LLM 客户端初始化成功")
            self.logger.debug(f"  模型: {collate_model_config.model_name}")
            self.logger.debug(f"  温度: {collate_model_config.temperature}")
            self.logger.debug(f"  最大tokens: {collate_model_config.max_tokens}")
        except Exception as e:
            self.logger.error(f"❌ Collate LLM 客户端初始化失败: {e}")
            self.logger.warning(f"⚠️ 将使用默认 LLM 客户端作为备用")
            self.collate_llm_client = self.llm_client

    def analyze_task(self, task: str) -> str:
        """
        分析任务 - 基于 MBTI 特质的结构化分析，输出 XML 格式并存入知识库
        
        执行流程：
        1. 调用 LLMToolClient 获取结构化分析结果（XML 格式）
        2. 解析 XML 提取多个知识点
        3. 为每个知识点添加 meta 信息（智能体ID、MBTI类型等）
        4. 使用 add_knowledge_tool 将知识存入向量数据库
        
        Args:
            task: 任务描述
            
        Returns:
            包含分析结果的字典：
            {
                'success': bool,  # 分析是否成功
                'knowledge_points_count': int,  # 提取的知识点数量
                'stored_count': int,  # 成功存储的知识点数量
                'summary': str,  # 一句话总结
                'knowledge_points': list  # 知识点列表
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        self.logger.info(f"🧠 {logger_name} 开始分析任务")
        self.logger.debug(f"  任务长度: {len(task)} 字符")
        
        # 检查必要组件
        if not self.llm_tool_client:
            self.logger.error(f"❌ {logger_name} LLMToolClient 未初始化")
            return {
                'success': False,
                'knowledge_points_count': 0,
                'stored_count': 0,
                'summary': 'LLM工具客户端未初始化',
                'knowledge_points': []
            }
        
        if not self.knowledge_manager:
            self.logger.warning(f"⚠️ {logger_name} 知识库管理器未初始化，将跳过知识存储")
        
        try:
            # 1. 加载提示词模板
            self.logger.debug(f"{logger_name} 加载提示词模板...")
            user_prompt_template = prompt_loader.load_function(
                mbti_type=self.mbti_type.lower(),
                function_name=task_name
            )
            
            # 2. 格式化提示词
            user_prompt = user_prompt_template.format(
                agent_id=self.agent_id,
                task=task
            )
            
            # 获取性格系统提示词
            system_prompt = self.get_personality_prompt()
            
            self.logger.debug(f"{logger_name} 提示词准备完成")
            self.logger.debug(f"  系统提示词长度: {len(system_prompt)} 字符")
            self.logger.debug(f"  用户提示词长度: {len(user_prompt)} 字符")
            
            # 3. 调用 LLM（不使用工具调用）
            self.logger.info(f"🤖 {logger_name} 调用大模型进行分析...")
            
            messages = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=self.config.model.model_name,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens
            )
            self.logger.debug(f"  LLM Response: {response}")
            # 提取响应文本
            response_text = response.get('content', '')
            
            if not response_text:
                self.logger.error(f"❌ {logger_name} 大模型返回空响应")
                return {
                    'success': False,
                    'knowledge_points_count': 0,
                    'stored_count': 0,
                    'summary': '模型无响应',
                    'knowledge_points': []
                }
            
            self.logger.info(f"✓ {logger_name} 大模型分析完成")
            self.logger.debug(f"  响应长度: {len(response_text)} 字符")
            
            # 4. 解析 XML 提取知识点
            self.logger.debug(f"{logger_name} 解析知识点...")
            from src.utils.helpers import parse_knowledge_points, build_knowledge_with_meta
            
            knowledge_points = parse_knowledge_points(response_text)
            
            if not knowledge_points:
                self.logger.warning(f"⚠️ {logger_name} 未能从响应中解析出知识点")
                self.logger.debug(f"  原始响应: {response_text}")
                return {
                    'success': False,
                    'knowledge_points_count': 0,
                    'stored_count': 0,
                    'summary': '未能解析出知识点',
                    'knowledge_points': [],
                    'raw_response': response_text
                }
            
            self.logger.info(f"✓ {logger_name} 成功解析 {len(knowledge_points)} 个知识点")
            
            # 5. 将所有分析点整合为一个完整的任务分析报告
            stored_count = 0
            if self.knowledge_manager and knowledge_points:
                self.logger.debug(f"  开始整合 {len(knowledge_points)} 个分析点为完整报告...")
                
                # 构建完整的任务分析报告
                report_parts = [f"【任务分析报告 - {self.agent_id}】\n"]
                report_parts.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_parts.append(f"MBTI类型: {self.mbti_type}")
                report_parts.append(f"分析维度: {len(knowledge_points)} 个\n")
                
                for idx, kp in enumerate(knowledge_points, 1):
                    title = kp.get('title', '未命名')
                    content = kp.get('content', '')
                    importance = kp.get('importance', 'medium')
                    
                    self.logger.debug(f"  整合分析点 {idx}/{len(knowledge_points)}: {title}")
                    
                    # 添加到报告中
                    report_parts.append(f"\n## {idx}. {title} (重要性: {importance})")
                    report_parts.append(content)
                
                # 组合成完整报告
                integrated_report = "\n".join(report_parts)
                
                # 构建知识数据（整合后的单条记录）
                knowledge_data = build_knowledge_with_meta(
                    knowledge_content=integrated_report,
                    agent_id=self.agent_id,
                    mbti_type=self.mbti_type,
                    task_id=self.current_task_id,
                    knowledge_type="task_analysis_report",
                    importance="high",  # 整合报告默认高重要性
                    additional_meta={
                        'report_type': 'task_analysis',
                        'analysis_count': len(knowledge_points),
                        'integrated': True
                    }
                )
                
                try:
                    self.logger.info(f"📦 存储整合的任务分析报告（{len(knowledge_points)} 个分析点）...")
                    result = add_knowledge_batch_tool(knowledge_list=[{
                        'knowledge': knowledge_data['content'],
                        'metadata': knowledge_data['metadata']
                    }])
                    result_dict = json.loads(result) if isinstance(result, str) else result
                    
                    if result_dict.get('success'):
                        stored_count = 1
                        self.logger.info(f"  ✓ 整合报告存储成功: 1 条记录")
                    else:
                        self.logger.error(f"  ✗ 整合报告存储失败: {result_dict.get('message')}")
                except Exception as e:
                    self.logger.error(f"  ✗ 整合报告存储异常: {e}")
            elif not self.knowledge_manager:
                self.logger.debug(f"  跳过存储（无知识库管理器）")
            
            self.logger.info(f"✅ {logger_name} 任务分析完成，成功存储 {stored_count}/{len(knowledge_points)} 个知识点")
            
            # 6. 生成摘要返回
            summary_parts = [f"【任务分析完成 - {self.mbti_type}】"]
            summary_parts.append(f"共提取 {len(knowledge_points)} 个关键分析点：")
            for idx, kp in enumerate(knowledge_points, 1):
                summary_parts.append(f"{idx}. {kp.get('title', '未命名')} (重要性: {kp.get('importance', 'medium')})")
            
            if stored_count > 0:
                summary_parts.append(f"\n已成功存储 {stored_count} 个知识点到向量知识库。")
            
            summary = "\n".join(summary_parts)
            
            # 7. 保存到讨论摘要（用于后续大纲生成）
            if self.discussion_summary_manager and knowledge_points:
                # 提取关键分析点作为摘要
                key_insights = [kp.get('title', '') for kp in knowledge_points[:3]]  # 取前3个最重要的
                summary_content = " | ".join(key_insights)
                self.discussion_summary_manager.add_summary(
                    action="analyze_task",
                    agent_id=self.agent_id,
                    content=summary_content
                )
                self.logger.debug(f"{logger_name} 已保存分析摘要到讨论记录")
            
            # 8. 构建返回结果
            result = {
                'success': True,
                'knowledge_points_count': len(knowledge_points),
                'stored_count': stored_count,
                'summary': summary,
                'knowledge_points': knowledge_points
            }
            
            self.logger.debug(f"{logger_name} 返回分析结果: {result}")
            
            return result
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ {logger_name} 提示词文件未找到: {e}")
            return {
                'success': False,
                'knowledge_points_count': 0,
                'stored_count': 0,
                'summary': f'提示词文件缺失 - {e}',
                'knowledge_points': []
            }
        except Exception as e:
            self.logger.error(f"❌ {logger_name} 分析任务时出错: {e}", exc_info=True)
            return {
                'success': False,
                'knowledge_points_count': 0,
                'stored_count': 0,
                'summary': str(e),
                'knowledge_points': []
            }

    def evaluate_solution(self, task: str, solution: str) -> Dict[str, Any]:
        """
        评估解决方案 - 基于 MBTI 特质的结构化评审，输出 XML 格式并存入知识库
        
        执行流程：
        1. 调用 LLMToolClient 获取结构化评审结果（XML 格式）
        2. 解析 XML 提取评审决策和多个评价点
        3. 为每个评价点添加 meta 信息（智能体ID、MBTI类型等）
        4. 使用 add_knowledge_tool 将评审知识存入向量数据库
        
        Args:
            task: 原始任务描述
            solution: 待评估的解决方案
            
        Returns:
            包含评审决策和评审程度的字典：
            {
                'approved': bool,  # 是否通过
                'approval_status': str,  # 通过/有条件通过/建议修改/拒绝
                'confidence_level': int,  # 0-100，评审信心程度
                'summary': str,  # 一句话总结
                'evaluation_points_count': int  # 评价点数量
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        self.logger.info(f"🔍 {logger_name} 开始评估解决方案")
        self.logger.debug(f"  任务长度: {len(task)} 字符")
        self.logger.debug(f"  方案长度: {len(solution)} 字符")
        
        # 检查必要组件
        if not self.llm_tool_client:
            self.logger.error(f"❌ {logger_name} LLMToolClient 未初始化")
            return {
                'approved': False,
                'approval_status': '系统错误',
                'confidence_level': 0,
                'summary': 'LLM工具客户端未初始化',
                'evaluation_points_count': 0
            }
        
        if not self.knowledge_manager:
            self.logger.warning(f"⚠️ {logger_name} 知识库管理器未初始化，将跳过知识存储")
        
        try:
            # 1. 加载提示词模板
            self.logger.debug(f"{logger_name} 加载提示词模板...")
            user_prompt_template = prompt_loader.load_function(
                mbti_type=self.mbti_type.lower(),
                function_name=task_name
            )
            
            # 2. 格式化提示词
            user_prompt = user_prompt_template.format(
                agent_id=self.agent_id,
                task=task,
                solution=solution
            )
            
            # 获取性格系统提示词
            system_prompt = self.get_personality_prompt()
            
            self.logger.debug(f"{logger_name} 提示词准备完成")
            self.logger.debug(f"  系统提示词长度: {len(system_prompt)} 字符")
            self.logger.debug(f"  用户提示词长度: {len(user_prompt)} 字符")
            
            # 3. 调用 LLM（不使用工具调用）
            self.logger.info(f"🤖 {logger_name} 调用大模型进行评审...")
            
            messages = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=self.config.model.model_name,
                temperature=self.config.model.temperature * 0.7,  # 评审时降低温度以获得更稳定的结果
                max_tokens=self.config.model.max_tokens
            )
            self.logger.debug(f"  LLM Response: {response}")
                                   
            # 提取响应文本
            response_text = response.get('content', '')
            
            if not response_text:
                self.logger.error(f"❌ {logger_name} 大模型返回空响应")
                return {
                    'approved': False,
                    'approval_status': '评审失败',
                    'confidence_level': 0,
                    'summary': '模型无响应',
                    'evaluation_points_count': 0
                }
            
            self.logger.info(f"✓ {logger_name} 大模型评审完成")
            self.logger.debug(f"  响应长度: {len(response_text)} 字符")
            
            # 4. 解析 XML 提取评审结果
            self.logger.debug(f"{logger_name} 解析评审结果...")
            from src.utils.helpers import parse_evaluation_result, build_knowledge_with_meta
            
            eval_result = parse_evaluation_result(response_text)
            
            if not eval_result.get('decision'):
                self.logger.warning(f"⚠️ {logger_name} 未能从响应中解析出评审决策")
                self.logger.debug(f"  原始响应: {response_text[:500]}...")
                return {
                    'approved': False,
                    'approval_status': '解析失败',
                    'confidence_level': 0,
                    'summary': '无法解析评审结果',
                    'evaluation_points_count': 0
                }
            
            decision = eval_result['decision']
            evaluation_points = eval_result['evaluation_points']
            
            self.logger.info(f"✓ {logger_name} 成功解析评审结果")
            self.logger.debug(f"  决策: {decision.get('approval_status')}")
            self.logger.debug(f"  信心程度: {decision.get('confidence_level')}")
            self.logger.debug(f"  评价点数量: {len(evaluation_points)}")
            
            # 5. 将所有评价点整合为一个完整的方案评审报告
            stored_count = 0
            if self.knowledge_manager and evaluation_points:
                self.logger.debug(f"  开始整合 {len(evaluation_points)} 个评价点为完整报告...")
                
                # 构建完整的方案评审报告
                report_parts = [f"【方案评审报告 - {self.agent_id}】\n"]
                report_parts.append(f"评审时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_parts.append(f"MBTI类型: {self.mbti_type}")
                report_parts.append(f"评审决策: {decision.get('approval_status', '未知')}")
                report_parts.append(f"信心程度: {decision.get('confidence_level', 0)}")
                report_parts.append(f"评价维度: {len(evaluation_points)} 个\n")
                
                # 添加评审决策摘要
                if decision.get('summary'):
                    report_parts.append(f"## 评审摘要\n{decision.get('summary')}\n")
                
                # 添加每个评价点
                for idx, ep in enumerate(evaluation_points, 1):
                    title = ep.get('title', '未命名')
                    content = ep.get('content', '')
                    dimension = ep.get('dimension', 'general')
                    importance = ep.get('importance', 'medium')
                    
                    self.logger.debug(f"  整合评价点 {idx}/{len(evaluation_points)}: {title}")
                    
                    report_parts.append(f"\n## {idx}. {title}")
                    report_parts.append(f"维度: {dimension} | 重要性: {importance}")
                    report_parts.append(content)
                
                # 组合成完整报告
                integrated_report = "\n".join(report_parts)
                
                # 构建知识数据（整合后的单条记录）
                knowledge_data = build_knowledge_with_meta(
                    knowledge_content=integrated_report,
                    agent_id=self.agent_id,
                    mbti_type=self.mbti_type,
                    task_id=self.current_task_id,
                    knowledge_type="solution_evaluation_report",
                    importance="high",  # 整合报告默认高重要性
                    additional_meta={
                        'report_type': 'solution_evaluation',
                        'approval_status': decision.get('approval_status'),
                        'confidence_level': decision.get('confidence_level'),
                        'evaluation_count': len(evaluation_points),
                        'integrated': True
                    }
                )
                
                try:
                    self.logger.info(f"📦 存储整合的方案评审报告（{len(evaluation_points)} 个评价点）...")
                    result = add_knowledge_batch_tool(knowledge_list=[{
                        'knowledge': knowledge_data['content'],
                        'metadata': knowledge_data['metadata']
                    }])
                    result_dict = json.loads(result) if isinstance(result, str) else result
                    
                    if result_dict.get('success'):
                        stored_count = 1
                        self.logger.info(f"  ✓ 整合报告存储成功: 1 条记录")
                    else:
                        self.logger.error(f"  ✗ 整合报告存储失败: {result_dict.get('message')}")
                except Exception as e:
                    self.logger.error(f"  ✗ 整合报告存储异常: {e}")
            elif not self.knowledge_manager:
                self.logger.debug(f"  跳过存储（无知识库管理器）")
            
            self.logger.info(f"✅ {logger_name} 评审完成，成功存储 {stored_count} 个知识点")
            
            # 7. 构建返回结果
            approval_status = decision.get('approval_status', '未知')
            approved = approval_status in ['通过', '有条件通过']
            
            # 8. 保存到讨论摘要
            if self.discussion_summary_manager:
                summary_content = decision.get('summary', '')[:200]  # 截取前200字符
                self.discussion_summary_manager.add_summary(
                    action="evaluate_solution",
                    agent_id=self.agent_id,
                    content=summary_content,
                    metadata={
                        "approval_status": approval_status,
                        "confidence_level": decision.get('confidence_level', 0)
                    }
                )
                self.logger.debug(f"{logger_name} 已保存评审摘要到讨论记录")
            
            # 9. 返回结果字典
            
            result = {
                'approved': approved,
                'approval_status': approval_status,
                'confidence_level': decision.get('confidence_level', 0),
                'summary': decision.get('summary', '无摘要'),
                'evaluation_points_count': len(evaluation_points),
                'evaluation_points': evaluation_points
            }
            
            self.logger.debug(f"{logger_name} 返回评审结果: {result}")
            
            return result
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ {logger_name} 提示词文件未找到: {e}")
            return {
                'approved': False,
                'approval_status': '配置错误',
                'confidence_level': 0,
                'summary': f'提示词文件缺失 - {e}',
                'evaluation_points_count': 0
            }
        except Exception as e:
            self.logger.error(f"❌ {logger_name} 评估方案时出错: {e}", exc_info=True)
            return {
                'approved': False,
                'approval_status': '系统错误',
                'confidence_level': 0,
                'summary': str(e),
                'evaluation_points_count': 0
            }
 

    def get_personality_prompt(self) -> str:
        """
        获取性格提示词
        每个MBTI智能体需要实现自己的性格描述
        
        Returns:
            性格提示词
        """
        return prompt_loader.load_personality(self.mbti_type.lower())

    def decide_team_preference(self, agents: List, task: str, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        MBTI 的组队偏好 - 基于第一轮讨论内容，从多个维度评估队友，给出组队意愿
        
        执行流程：
        1. 整理其他智能体的任务分析结果
        2. 调用 LLMToolClient 获取结构化评估结果（XML 格式）
        3. 解析 XML 提取各候选人的评分和偏好程度
        4. 将程度词映射为分数，返回结构化结果
        
        Args:
            agents: 所有候选智能体列表
            task: 任务上下文描述
            analyses: 第一轮讨论的分析内容 {agent_id: analysis_result}
        
        Returns:
            包含组队偏好的详细评估结果：
            {
                'success': bool,
                'overall_strategy': str,
                'candidates': [
                    {
                        'agent_id': str,
                        'scores': {
                            'collaboration_potential': float,
                            'capability_complement': float,
                            'communication_fit': float,
                            'growth_value': float,
                            'total_score': float
                        },
                        'preference_level': str,
                        'preference_score': float,
                        'brief_comment': str
                    }
                ],
                'recommendations': str,
                'suggestions': [str]
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        self.logger.info(f"🤝 {logger_name} 开始计算组队意愿")
        self.logger.debug(f"  候选智能体数量: {len(agents)}")
        self.logger.debug(f"  分析结果数量: {len(analyses)}")
        
        # 检查必要组件
        if not self.llm_tool_client:
            self.logger.error(f"❌ {logger_name} LLMToolClient 未初始化")
            return {
                'success': False,
                'overall_strategy': '',
                'candidates': [],
                'recommendations': 'LLM工具客户端未初始化',
                'suggestions': []
            }

        try:
            # 1. 整理第一轮讨论内容
            self.logger.debug(f"{logger_name} 整理第一轮讨论内容...")
            round_1_discussion = ''
            
            if not analyses:
                round_1_discussion = "【暂无第一轮讨论内容】"
                self.logger.warning(f"{logger_name} 未收到任何分析结果")
            else:
                for agent_id, analysis in analyses.items():
                    # 跳过自己
                    if agent_id == self.agent_id:
                        continue
                    
                    # 查找对应的智能体
                    now_agent = None
                    for agent in agents:
                        if agent_id == agent.agent_id:
                            now_agent = agent
                            break
                    
                    if now_agent is None:
                        self.logger.warning(f"{logger_name} 未找到智能体: {agent_id}")
                        continue
                    
                    # 格式化讨论内容（使用系统提示词模板）
                    try:
                        # 如果 analysis 是字典（analyze_task 返回的结构化结果）
                        if isinstance(analysis, dict):
                            # 构建知识点列表字符串
                            knowledge_points = analysis.get('knowledge_points', [])
                            knowledge_points_list = ""
                            if knowledge_points:
                                for idx, kp in enumerate(knowledge_points, 1):
                                    title = kp.get('title', '未命名')
                                    importance = kp.get('importance', 'medium')
                                    content_preview = kp.get('content', '')[:100]
                                    knowledge_points_list += f"  {idx}. {title} [重要性: {importance}]"
                                    if content_preview:
                                        knowledge_points_list += f"\n     {content_preview}..."
                                    knowledge_points_list += "\n"
                            else:
                                knowledge_points_list = "  暂无具体知识点"
                            
                            # 加载字典格式模板
                            template = prompt_loader.load_system_prompt(
                                "decide_team_preference_discussion_format_dict"
                            )
                            formatted_discussion = template.format(
                                agent_id=agent_id,
                                mbti_type=now_agent.mbti_type,
                                status='成功' if analysis.get('success') else '失败',
                                summary=analysis.get('summary', '无摘要'),
                                points_count=analysis.get('knowledge_points_count', 0),
                                knowledge_points_list=knowledge_points_list.strip()
                            )
                        else:
                            # 加载字符串格式模板
                            template = prompt_loader.load_system_prompt(
                                "decide_team_preference_discussion_format_str"
                            )
                            formatted_discussion = template.format(
                                agent_id=agent_id,
                                mbti_type=now_agent.mbti_type,
                                analysis=analysis
                            )
                        
                        round_1_discussion += formatted_discussion + "\n"
                        
                    except Exception as e:
                        self.logger.error(f"{logger_name} 格式化讨论内容失败: {e}")
                        # 降级处理：直接拼接
                        round_1_discussion += f"\n【智能体 {agent_id}】\n{analysis}\n\n"
            
            self.logger.debug(f"{logger_name} 讨论内容长度: {len(round_1_discussion)} 字符")
            
            # 2. 加载提示词模板
            self.logger.debug(f"{logger_name} 加载提示词模板...")
            user_prompt_template = prompt_loader.load_function(
                mbti_type=self.mbti_type.lower(),
                function_name=task_name
            )
            
            # 3. 格式化提示词
            user_prompt = user_prompt_template.format(
                agent_id=self.agent_id,
                task=task,
                round_1_discussion=round_1_discussion
            )
            
            # 获取性格系统提示词
            system_prompt = self.get_personality_prompt()
            
            self.logger.debug(f"{logger_name} 提示词准备完成")
            self.logger.debug(f"  系统提示词长度: {len(system_prompt)} 字符:{system_prompt}")
            self.logger.debug(f"  用户提示词长度: {len(user_prompt)} 字符:{user_prompt}")
            
            # 4. 调用 LLM
            self.logger.info(f"🤖 {logger_name} 调用大模型进行组队评估...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=self.config.model.model_name,
                temperature=self.config.model.temperature * 0.8,  # 降低温度以获得更一致的评估
                max_tokens=self.config.model.max_tokens
            )
            self.logger.debug(f"  LLM Response: {response}")
                       
            # 提取响应文本
            response_text = response.get('content', '')
            
            if not response_text:
                self.logger.error(f"❌ {logger_name} 大模型返回空响应")
                return {
                    'success': False,
                    'overall_strategy': '',
                    'candidates': [],
                    'recommendations': '模型无响应',
                    'suggestions': []
                }
            
            self.logger.info(f"✓ {logger_name} 大模型评估完成")
            self.logger.debug(f"  响应长度: {len(response_text)} 字符")
            
            # 5. 解析 XML 提取评估结果
            self.logger.debug(f"{logger_name} 解析组队偏好结果...")
            from src.utils.helpers import parse_team_preference_result
            
            preference_result = parse_team_preference_result(response_text)
            
            if not preference_result.get('candidates'):
                self.logger.warning(f"⚠️ {logger_name} 未能解析出候选人评估")
                self.logger.debug(f"  原始响应: {response_text[:500]}...")
                return {
                    'success': False,
                    'overall_strategy': preference_result.get('overall_strategy', ''),
                    'candidates': [],
                    'recommendations': preference_result.get('recommendations', '解析失败'),
                    'suggestions': preference_result.get('suggestions', []),
                    'raw_response': response_text
                }
            
            self.logger.info(f"✓ {logger_name} 成功解析组队偏好")
            self.logger.debug(f"  候选人数量: {len(preference_result['candidates'])}")
            self.logger.debug(f"  建议数量: {len(preference_result.get('suggestions', []))}")
            
            # 6. 构建返回结果
            result = {
                'success': True,
                'overall_strategy': preference_result.get('overall_strategy', ''),
                'candidates': preference_result.get('candidates', []),
                'recommendations': preference_result.get('recommendations', ''),
                'suggestions': preference_result.get('suggestions', [])
            }
            
            # 记录评估摘要
            if result['candidates']:
                self.logger.info(f"📊 {logger_name} 组队偏好评估摘要:")
                for candidate in sorted(result['candidates'], 
                                       key=lambda x: x['preference_score'], 
                                       reverse=True):
                    self.logger.info(
                        f"  {candidate['agent_id']}: "
                        f"{candidate['preference_level']} "
                        f"(总分: {candidate['scores']['total_score']}, "
                        f"偏好分: {candidate['preference_score']})"
                    )
            
            # 7. 将组队偏好评估结果存入知识库（整合为完整报告）
            if self.knowledge_manager and result['success']:
                self.logger.debug(f"{logger_name} 开始整合组队偏好评估结果为完整报告...")
                from src.utils.helpers import build_knowledge_with_meta
                
                # 构建完整的组队策略报告
                report_parts = [f"【组队策略报告 - {self.agent_id}】\n"]
                report_parts.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_parts.append(f"MBTI类型: {self.mbti_type}")
                report_parts.append(f"候选人数: {len(result['candidates'])} 人")
                report_parts.append(f"任务描述: {task[:200]}...\n")
                
                # 添加总体策略
                if result.get('overall_strategy'):
                    report_parts.append(f"## 总体策略\n{result['overall_strategy']}\n")
                
                # 添加每个候选人的评估结果
                if result['candidates']:
                    report_parts.append("## 候选人评估")
                    for idx, candidate in enumerate(sorted(result['candidates'], 
                                                           key=lambda x: x['preference_score'], 
                                                           reverse=True), 1):
                        candidate_id = candidate['agent_id']
                        preference_level = candidate['preference_level']
                        scores = candidate['scores']
                        comment = candidate.get('brief_comment', '')
                        
                        report_parts.append(f"\n### {idx}. {candidate_id}")
                        report_parts.append(f"偏好程度: {preference_level}")
                        report_parts.append(f"综合得分: {scores['total_score']}")
                        
                        # 动态构建评估维度内容
                        dimensions_text = []
                        for key, value in scores.items():
                            if key != 'total_score':
                                readable_key = key.replace('_', ' ').title()
                                dimensions_text.append(f"- {readable_key}: {value}")
                        if dimensions_text:
                            report_parts.append("评估维度:")
                            report_parts.extend(dimensions_text)
                        
                        if comment:
                            report_parts.append(f"评价: {comment}")
                
                # 添加推荐理由
                if result.get('recommendations'):
                    report_parts.append(f"\n## 推荐理由\n{result['recommendations']}")
                
                # 添加补充建议
                if result.get('suggestions'):
                    report_parts.append("\n## 补充建议")
                    for idx, suggestion in enumerate(result['suggestions'], 1):
                        report_parts.append(f"{idx}. {suggestion}")
                
                # 组合成完整报告
                integrated_report = "\n".join(report_parts)
                
                # 构建知识数据（整合后的单条记录）
                knowledge_data = build_knowledge_with_meta(
                    knowledge_content=integrated_report,
                    agent_id=self.agent_id,
                    mbti_type=self.mbti_type,
                    task_id=self.current_task_id,
                    knowledge_type="team_preference_report",
                    importance="high",  # 整合报告默认高重要性
                    additional_meta={
                        'report_type': 'team_preference',
                        'candidates_count': len(result['candidates']),
                        'integrated': True
                    }
                )
                
                # 存储整合后的单条报告
                stored_count = 0
                try:
                    self.logger.info(f"📦 存储整合的组队策略报告（{len(result['candidates'])} 个候选人）...")
                    result_json = add_knowledge_batch_tool(knowledge_list=[{
                        'knowledge': knowledge_data['content'],
                        'metadata': knowledge_data['metadata']
                    }])
                    result_dict = json.loads(result_json) if isinstance(result_json, str) else result_json
                    
                    if result_dict.get('success'):
                        stored_count = 1
                        self.logger.info(f"  ✓ 整合报告存储成功: 1 条记录")
                    else:
                        self.logger.error(f"  ✗ 整合报告存储失败: {result_dict.get('message')}")
                except Exception as e:
                    self.logger.error(f"  ✗ 整合报告存储异常: {e}")
                
                self.logger.info(f"✅ {logger_name} 组队评估存储完成: {stored_count} 个知识点")
            
            # 8. 保存到讨论摘要
            if self.discussion_summary_manager and result.get('overall_strategy'):
                strategy_content = result['overall_strategy'][:200]  # 截取前200字符
                self.discussion_summary_manager.add_summary(
                    action="decide_team_preference",
                    agent_id=self.agent_id,
                    content=strategy_content,
                    metadata={
                        "candidates_count": len(result.get('candidates', []))
                    }
                )
                self.logger.debug(f"{logger_name} 已保存组队偏好摘要到讨论记录")
            
            self.logger.info(f"✅ {logger_name} 组队意愿计算完成")
            
            return result
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ {logger_name} 提示词文件未找到: {e}")
            return {
                'success': False,
                'overall_strategy': '',
                'candidates': [],
                'recommendations': f'提示词文件缺失 - {e}',
                'suggestions': []
            }
        except Exception as e:
            self.logger.error(f"❌ {logger_name} 组队意愿计算失败: {e}", exc_info=True)
            return {
                'success': False,
                'overall_strategy': '',
                'candidates': [],
                'recommendations': str(e),
                'suggestions': []
            }

    def update_speaking_intention(self,intention_score, intention_reason):
        """
        监听消息,更新发言意愿。每次有人发言之后，其他人都要听的。
        
        Args:
            message: 消息对象
        """
        # 分析消息内容,更新发言意愿
        self.speaking_intention.intention_score = intention_score
        self.speaking_intention.context = intention_reason
        self.speaking_intention.timestamp = datetime.now()
        
        self.logger.info(f"发言意愿更新: score={intention_score:.2f}, threshold={self.speaking_intention.threshold}")
        
        # 更新到发言队列。
        self.speaking_queue.update_intention(self.agent_id, self.speaking_intention)
             
    def speak(self, task_id: str, team_id: str, content: str, 
              message_type: MessageType = MessageType.DISCUSSION,
              round_stage: RoundStage = RoundStage.ROUND_1_DISCUSSION) -> Message:
        """
        发言。原本设计用来记录发言，现在在子类里面调用，用来记录思维过程。
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            content: 发言内容
            message_type: 消息类型
            round_stage: 轮次阶段
            
        Returns:
            创建的消息对象
        """
        self.logger.info(f"💬 准备发言")
        self.logger.debug(f"  任务ID: {task_id}")
        self.logger.debug(f"  团队ID: {team_id}")
        self.logger.debug(f"  消息类型: {message_type}")
        self.logger.debug(f"  轮次阶段: {round_stage}")
        self.logger.debug(f"  内容长度: {len(content)} 字符")
        
        sequence = self.speaking_queue.get_next_sequence()
        self.logger.debug(f"  发言序号: {sequence}")
        
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            team_id=team_id,
            round_stage=round_stage,
            sequence=sequence,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        
        self.logger.debug(f"  消息ID: {message.message_id}")
        
        # 保存消息
        self.message_manager.save_message(task_id, message)
        self.logger.info(f"✅ 消息已保存")
        
        # 重置发言意愿
        self.speaking_intention.intention_score = 0.0 # 重置为0，然后自己也会听自己发言，更新意愿。所以允许存在连续发言的可能性。
        self.speaking_queue.update_intention(self.agent_id, self.speaking_intention)
        self.logger.debug("发言意愿已重置")
        
        return message
    
    def get_team_messages(self, task_id: str, team_id: str) -> List[Message]:
        """
        获取团队消息
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            
        Returns:
            消息列表
        """
        return self.message_manager.get_messages_by_team(task_id, team_id)
    
    def _recall_memory_by_round(self, task_id: str =None, stages: List[RoundStage] =[], top_k: int = 3) -> List[Message]:
        """
        召回记忆 - 基于消息系统的统一记忆管理
        
        从消息系统中获取指定轮次阶段的消息,按时间戳排序,返回最近的k条消息。 越新的消息排在越前面。
        
        Args:
            task_id: 任务ID
            stages: 轮次阶段列表
            top_k: 返回最近k条消息,默认为3
            
        Returns:
            消息列表,按时间从新到旧排序
        """
        self.logger.debug(f"召回记忆 by round: task_id={task_id}, round_stages={stages}, top_k={top_k}")
        task_id = task_id or self.current_task_id
        
        # 获取指定轮次阶段的消息
        messages_by_round = []
        for stage in stages:
            msgs = self.message_manager.get_messages_by_round(task_id, stage)
            messages_by_round.extend(msgs)
        
        if not messages_by_round:
            self.logger.debug(f"任务 {task_id} 中暂无指定轮次阶段的消息")
            return []
        
        # 按时间戳排序,最新的在前
        sorted_messages = sorted(
            messages_by_round, 
            key=lambda m: m.timestamp, 
            reverse=True
        )
        
        recent_messages = []
        
        for msg in sorted_messages:
            # 获取最近的top_k条
            recent_messages.append(msg)
            if len(recent_messages) >= top_k:
                break
        self.logger.debug(f"从 {len(messages_by_round)} 条消息中召回最近 {len(recent_messages)} 条")
        
        return recent_messages
    
    def get_visible_messages(self, task_id: str) -> List[Message]:
        """
        获取自己可见的所有消息
        
        Args:
            task_id: 任务ID
            
        Returns:
            消息列表
        """
        return self.message_manager.get_messages_by_agent(task_id, self.agent_id)
    
    def _recall_memory(self, task_id: Optional[str] = None, top_k: int = 3,stages: Optional[List[RoundStage]] = None) -> List[Message]:
        """
        召回记忆 - 基于消息系统的统一记忆管理
        
        从消息系统中获取该智能体可见的所有消息,按时间戳排序,返回最近的k条消息。 越新的消息排在越前面。
        
        Args:
            task_id: 任务ID,如果为None则使用当前任务ID
            top_k: 返回最近k条消息,默认为6
            
        Returns:
            消息列表,按时间从新到旧排序
        """
        # 使用当前任务ID或传入的任务ID
        task_id = task_id or self.current_task_id
        self.logger.debug(f"召回记忆: task_id={task_id}, top_k={top_k}")
        if not task_id:
            self.logger.warning("无法召回记忆: 未指定任务ID且当前任务ID为空")
            return []
        
        # 获取可见消息
        visible_messages = self.get_visible_messages(task_id)
        
        if not visible_messages:
            self.logger.debug(f"任务 {task_id} 中暂无可见消息")
            return []
        
        # 按时间戳排序,最新的在前
        sorted_messages = sorted(
            visible_messages, 
            key=lambda m: m.timestamp, 
            reverse=True
        )
        
        stages = stages or []
        recent_messages = []
        
        for msg in sorted_messages:
            if stages and msg.round_stage not in stages:
                continue
            # 获取最近的top_k条
            recent_messages.append(msg)
            if len(recent_messages) >= top_k:
                break
        self.logger.debug(f"从 {len(visible_messages)} 条消息中召回最近 {len(recent_messages)} 条")
        
        return recent_messages
    
    def _collate_conversation_history(self, recent_messages: List[Message], task: str) -> List[Message]:
        """
        整理对话历史 - 逐步摘要关键信息
        
        从后往前遍历消息列表，每次结合前两轮对话进行上下文感知的摘要，
        提取关键信息、引用、数据等，减少token消耗同时保留重要内容。
        
        Args:
            recent_messages: 原始消息列表（按时间从新到旧排序）
            task: 任务描述
            
        Returns:
            整理后的消息列表（保持原顺序和数据类型）
        """
        task_name = inspect.currentframe().f_code.co_name
        self.logger.info(f"📝 开始整理对话历史，共 {len(recent_messages)} 条消息")
        
        if not recent_messages:
            self.logger.debug("消息列表为空，跳过整理")
            return recent_messages
        
        # 如果消息数量很少，不需要摘要
        if len(recent_messages) <= 3:
            self.logger.debug("消息数量较少，无需整理")
            return recent_messages
        
        try:
            collated_messages = []
            
            # 从后往前遍历（越往前越新）
            for i in range(len(recent_messages) - 1, -1, -1):
                current_msg = recent_messages[i]
                
                # 获取前两轮对话（在列表中是索引更小的位置）
                previous_context = []
                for j in range(max(0, i - 1), i):
                    prev_msg = recent_messages[j]
                    # 检查是否同属一个团队（表示是连续讨论）
                    if prev_msg.team_id == current_msg.team_id:
                        previous_context.append(prev_msg)
                
                # 构建前两轮对话的上下文描述
                if previous_context:
                    context_desc = "【同团队的前续对话】\n"
                    for idx, prev_msg in enumerate(previous_context, 1):
                        context_desc += f"\n**对话{idx}** (来自 {prev_msg.sender_id}):\n{prev_msg.content}\n"
                else:
                    context_desc = "[无前续对话或不在同一团队]"
                
                # 调用LLM整理当前消息
                self.logger.debug(f"整理第 {len(recent_messages) - i}/{len(recent_messages)} 条消息 (ID: {current_msg.message_id})")
                
                prompt_template = prompt_loader.load_system_prompt(task_name)
                prompt = prompt_template.format(
                    task=task,
                    current_sender=current_msg.sender_id,
                    current_team=current_msg.team_id,
                    current_type=current_msg.message_type,
                    current_stage=current_msg.round_stage,
                    current_time=current_msg.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    current_content=current_msg.content,
                    previous_context=context_desc
                )
                
                collated_content = call_llm(
                    llm_client=self.llm_client,
                    model_name=self.config.model.model_name,
                    user_prompt=prompt,
                    temperature=0.3,  # 低温度保证信息提取的准确性
                    max_tokens=self.config.model.max_tokens,
                    logger=self.logger,
                    logger_name=f"【{self.agent_id};{task_name}】"
                )
                
                # 提取整理后的内容
                extracted = get_tag_content(collated_content, tag="ANSWER", is_json=False)
                collated_text = extracted.get("content", "").strip()
                
                citation = get_tag_content(current_msg.content, tag="CITATION", is_json=False)
                result = citation.get("result", "").strip()
                if not result:
                    self.logger.debug("当前消息无引用内容")
                collated_content += "\n\n" + result
                
                # 如果提取失败或为空，保留原内容
                if not collated_text:
                    self.logger.warning(f"消息整理失败，保留原内容 (ID: {current_msg.message_id})")
                    collated_text = current_msg.content
                
                self.logger.debug(f"="*80)
                self.logger.debug(f"【拼接内容】\n{collated_content}\n")
                self.logger.debug(f"="*80)
                # 创建新的消息对象，保留原有元数据，只更新内容
                collated_msg = Message(
                    message_id=current_msg.message_id,
                    sender_id=current_msg.sender_id,
                    team_id=current_msg.team_id,
                    round_stage=current_msg.round_stage,
                    sequence=current_msg.sequence,
                    message_type=current_msg.message_type,
                    content=collated_content,
                    timestamp=current_msg.timestamp
                )
                
                # 插入到列表开头（因为我们是从后往前遍历的）
                collated_messages.insert(0, collated_msg)
                
                self.logger.debug(f"✅ 消息整理完成：原长度 {len(current_msg.content)} → 新长度 {len(collated_text)}")
            
            self.logger.info(f"✅ 对话历史整理完成，共处理 {len(collated_messages)} 条消息")
            return collated_messages
            
        except Exception as e:
            self.logger.error(f"❌ 整理对话历史时出错: {e}", exc_info=True)
            self.logger.warning("回退到使用原始消息列表")
            return recent_messages
    
    def _format_conversation_history(self, messages: List[Message]) -> str:
        """
        格式化对话历史为字符串。
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的对话历史字符串
        """
        task_name = inspect.currentframe().f_code.co_name
        
        template = prompt_loader.load_system_prompt(task_name)
        
        history = ""
        
        for msg in messages:  # 从旧到新构建历史
            sender_id = msg.sender_id
            if sender_id == self.agent_id:
                sender_id += " (你自己)"
            current_message = template.format(
                sender_id=sender_id,
                team_id=msg.team_id,
                times=msg.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                message_type=msg.message_type,
                round_stage=msg.round_stage,
                content=msg.content
            )
            
            history += current_message + "\n"
            
        if not history:
            history = "[暂无历史对话]"
        
        return history.strip()
     
    ## generate 方法相关实现
    def init_representative_role(self)->str:
        '''
        初始化代表角色
        '''
        task_name = inspect.currentframe().f_code.co_name
        self.logger.info(f"🎭 {self.agent_id} 开始初始化代表角色...")
        representative_prompt_template = prompt_loader.load_system_prompt(task_name)
        representative_prompt = representative_prompt_template.format(
            team_id=self.representative_of_team,
            members=','.join(self.representative_of_members) if self.representative_of_members else '无成员信息'
        )
        
        return representative_prompt

    def generate(self, task: str, 
                        round_index = None,
                        max_rounds = None
                        ) -> Dict[str, Any]:
        """
        生成解决方案 - 基于 MBTI 特质的结构化生成，输出 XML 格式并存入知识库
        
        执行流程：
        1. 从知识库检索相关历史知识（混合策略：任务+意图检索 + 最近消息）
        2. 构建提示词，鼓励使用网页搜索工具
        3. 调用 LLMToolClient.call_with_tools 生成结构化内容
        4. 解析 XML 输出，提取思考过程、生成内容、知识点
        5. 为每个知识点添加 meta 信息并存储到知识库
        
        Args:
            task: 任务描述
            round_index: 当前讨论轮次
            max_rounds: 最大讨论轮次

        Returns:
            包含生成结果的字典：
            {
                'main_solution': str,  # 主要解决方案
                'thinking_summary': str,  # 思考过程摘要
                'team_collaboration': str,  # 团队协作建议
                'knowledge_points_count': int  # 存储的知识点数量
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        self.logger.info(f"🧠 {logger_name} 开始生成解决方案")
        self.logger.debug(f"  任务长度: {len(task)} 字符")
        
        # 检查必要组件
        if not self.llm_tool_client:
            error_msg = "LLMToolClient 未初始化，无法生成结构化方案"
            self.logger.error(f"❌ {error_msg}")
            return {
                'main_solution': f"【生成失败】{error_msg}",
                'thinking_summary': '',
                'team_collaboration': '',
                'knowledge_points_count': 0
            }
        
        if not self.knowledge_manager:
            self.logger.warning("⚠️ 知识库管理器未初始化，将跳过知识存储")
        
        try:
            # ========== 1. 混合策略检索相关历史知识 ==========
            self.logger.debug(f"{logger_name} 开始检索相关历史知识...")
            
            # 策略1: 使用任务+当前意图作为检索查询
            intention_context = self.speaking_intention.context if self.speaking_intention and self.speaking_intention.context else ""
            
            if intention_context:
                search_query = intention_context
            else:
                search_query = f"{task[:200]} {intention_context[:100]}" if intention_context else task[:200]
            
            self.logger.debug(f"  构建检索查询: {search_query}")
            
            # 使用结构化的知识检索方法（参考 collate 函数）
            knowledge_list = []
            if self.knowledge_manager:
                knowledge_list = self._retrieve_knowledge_for_generate(
                    task=search_query,
                    top_k=6
                )
                self.logger.debug(f"  检索到 {len(knowledge_list)} 条相关历史知识")
            
            # 使用统一的格式化方法（区分可引用和不可引用的知识）
            relevant_knowledge = self._format_knowledge_for_prompt(knowledge_list)
            
            # 策略2: 获取最近对话历史（保留连贯性）
            stages = [RoundStage.INITIAL_SOLUTION,
                     RoundStage.ROUND_1_DISCUSSION,
                     RoundStage.ROUND_3_TEAM_DISCUSSION,
                     RoundStage.ROUND_4_FINAL_DISCUSSION]
            recent_messages = self._recall_memory(top_k=1, stages=stages)
            conversation_history = self._format_conversation_history(recent_messages)
            
            # ========== 2. 构建提示词 ==========
            self.logger.debug(f"{logger_name} 构建提示词...")
            
            round_prompt = ""
            if round_index is not None and max_rounds is not None:
                round_prompt = f'''【当前讨论轮次】\n这是第 {round_index} 轮讨论，总共 {max_rounds} 轮，需要尽可能在最大轮次之前跟队友达成一致方案。\n'''
            
            user_prompt_template = prompt_loader.load_function(mbti_type=self.mbti_type, function_name=task_name)
            user_prompt = user_prompt_template.format(
                round_prompt=round_prompt,
                agent_id=self.agent_id,
                task=task,
                representative_prompt=self.init_representative_role() if self.representative_of_team else "",
                intention_reason=intention_context if intention_context else "暂无意图上下文",
                attitude_reason=self.attitude_reason if hasattr(self, 'attitude_reason') and self.attitude_reason else "暂无态度信息",
                conversation_history=conversation_history,
                relevant_knowledge=relevant_knowledge
            )
            
            system_prompt = self.get_personality_prompt()
            
            self.logger.debug(f"  系统提示词长度: {len(system_prompt)} 字符")
            self.logger.debug(f"  用户提示词长度: {len(user_prompt)} 字符:\n{user_prompt}")
            
            # ========== 3. 调用 LLMToolClient 生成结构化内容 ==========
            self.logger.info(f"🤖 {logger_name} 调用大模型进行结构化生成...")
            
            messages = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 注册网页搜索工具（如果还没注册）
            if 'web_search_tool' not in self.llm_tool_client.get_registered_tools():
                self.llm_tool_client._register_tools_from_list([web_search_tool])
                self.logger.debug("  已注册 web_search_tool 工具")
            
            response = self.llm_tool_client.call_with_tools(
                messages=messages,
                model_name=self.config.model.model_name,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                max_tool_iterations=5  # 允许多次工具调用
            )
            self.logger.debug(f"{logger_name} 大模型调用完成")
            self.logger.debug(f"  LLM Response: {response}")
            # 循环打印Response内容
              
            # call_with_tools 返回的键是 'content'，不是 'message'
            response_text = response.get('content', '')
            tool_call_records = response.get('tool_calls', [])
            
            # 统计工具调用次数
            web_search_count = sum(1 for tool in tool_call_records 
                                  if (tool.get('name') or tool.get('tool_name')) == 'web_search_tool')
            
            # ========== 3.5. 自动提取网页搜索结果中的知识点（批量优化版）==========
            web_search_knowledge_count = 0
            if self.knowledge_manager and tool_call_records:
                self.logger.debug(f"{logger_name} 检查工具调用记录，提取搜索结果知识...")
                
                # 收集所有待添加的知识点（批量处理）
                knowledge_batch = []
                
                for tool_record in tool_call_records:
                    # 兼容两种键名：'name' 和 'tool_name'
                    tool_name = tool_record.get('name') or tool_record.get('tool_name')
                    if tool_name == 'web_search_tool':
                        # 兼容两种键名：'result' 和 'tool_output'
                        tool_output = tool_record.get('result') or tool_record.get('tool_output', '')
                        
                        # 解析 XML 格式的搜索结果
                        if '<SEARCH_RESULTS>' in tool_output and '<RESULT' in tool_output:
                            self.logger.debug(f"  发现网页搜索结果，开始提取知识...")
                            
                            # 提取所有 <RESULT> 块
                            import re
                            from ..utils.helpers import extract_xml_tag_content, extract_all_xml_blocks
                            
                            results = extract_all_xml_blocks(tool_output, 'RESULT')
                            self.logger.debug(f"  提取到 {len(results)} 条搜索结果")
                            
                            for result_idx, result_xml in enumerate(results, 1):
                                # 提取关键信息
                                title = extract_xml_tag_content(result_xml, 'TITLE')
                                url = extract_xml_tag_content(result_xml, 'URL')
                                content_block = extract_xml_tag_content(result_xml, 'CONTENT')
                                description = extract_xml_tag_content(result_xml, 'DESCRIPTION')
                                
                                if not (title and content_block):
                                    continue
                                
                                # 收集description作为独立知识点（如果有且非空）
                                if description and description.strip():
                                    description_text = f"{title}\n{description}"
                                    knowledge_batch.append({
                                        'knowledge': description_text,
                                        'metadata': {
                                            'agent_id': self.agent_id,
                                            'mbti_type': self.mbti_type,
                                            'knowledge_type': 'web_search_overview',
                                            'importance': 'high',
                                            'timestamp': datetime.now().isoformat(),
                                            'task_id': self.current_task_id or 'unknown',
                                            'source_url': url or 'unknown',
                                            'source_title': title or 'unknown',
                                            'result_index': result_idx,
                                            'extraction_method': 'description_extract',
                                            'round_index': round_index if round_index is not None else -1
                                        }
                                    })
                                    self.logger.debug(f"    收集概述{result_idx}: {description[:50]}...")
                                
                                # 提取CONTENT中的所有CHUNK标签
                                chunks = extract_all_xml_blocks(content_block, 'CHUNK')
                                
                                if not chunks:
                                    # 兜底：如果没有CHUNK标签，将整个content作为一个知识点
                                    self.logger.debug(f"  结果{result_idx}未包含CHUNK标签，使用整体内容")
                                    chunks = [content_block[:1000]]  # 限制长度
                                
                                self.logger.debug(f"  结果{result_idx}包含 {len(chunks)} 个知识块")
                                
                                # 收集所有CHUNK到批处理列表
                                for chunk_idx, chunk_xml in enumerate(chunks, 1):
                                    # 提取CHUNK的属性和内容
                                    import re
                                    # 提取相似度和原始位置
                                    similarity_match = re.search(r'similarity="([0-9.]+)"', chunk_xml)
                                    position_match = re.search(r'original_position="(\d+)"', chunk_xml)
                                    
                                    similarity = similarity_match.group(1) if similarity_match else '0.0'
                                    original_position = position_match.group(1) if position_match else '0'
                                    
                                    # 提取chunk文本内容（去除标签）
                                    chunk_text = extract_xml_tag_content(chunk_xml, 'CHUNK')
                                    if not chunk_text:
                                        continue
                                    
                                    # 构建知识点
                                    knowledge_text = f"{title}\n{chunk_text}"
                                    
                                    # 加入批处理列表
                                    knowledge_batch.append({
                                        'knowledge': knowledge_text,
                                        'metadata': {
                                            'agent_id': self.agent_id,
                                            'mbti_type': self.mbti_type,
                                            'knowledge_type': 'web_search_chunk',
                                            'importance': 'high',
                                            'timestamp': datetime.now().isoformat(),
                                            'task_id': self.current_task_id or 'unknown',
                                            'source_url': url or 'unknown',
                                            'source_title': title or 'unknown',
                                            'similarity': similarity,
                                            'original_position': original_position,
                                            'chunk_index': chunk_idx,
                                            'result_index': result_idx,
                                            'extraction_method': 'semantic_chunk',
                                            'round_index': round_index if round_index is not None else -1
                                        }
                                    })
                                    self.logger.debug(f"    收集知识块{result_idx}-{chunk_idx} (相似度:{similarity})")
                
                # 批量存储所有知识点到知识库
                if knowledge_batch:
                    try:
                        self.logger.info(f"📦 {logger_name} 批量存储 {len(knowledge_batch)} 个搜索知识块...")
                        from ..tools.system.knowledgemanager import add_knowledge_batch_tool
                        result = add_knowledge_batch_tool(knowledge_list=knowledge_batch)
                        result_dict = json.loads(result) if isinstance(result, str) else result
                        
                        if result_dict.get('success'):
                            web_search_knowledge_count = result_dict.get('count', 0)
                            self.logger.info(f"✅ {logger_name} 批量存储成功: {web_search_knowledge_count} 个搜索知识块")
                        else:
                            self.logger.error(f"  ✗ 批量存储失败: {result_dict.get('message')}")
                    except Exception as e:
                        self.logger.error(f"  ✗ 批量存储异常: {e}")
                elif tool_call_records:
                    self.logger.debug(f"  未提取到有效的搜索知识块")
            
            # ========== 3.6. 保存搜索查询到讨论摘要 ==========
            if self.discussion_summary_manager and tool_call_records:
                search_queries = []
                for tool_record in tool_call_records:
                    tool_name = tool_record.get('name') or tool_record.get('tool_name')
                    if tool_name == 'web_search_tool':
                        # 提取 search_query 参数
                        arguments = tool_record.get('arguments', {})
                        
                        # 兼容不同格式：可能是字典或JSON字符串
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except Exception as e:
                                self.logger.debug(f"  ⚠️ 无法解析arguments: {arguments}")
                                continue
                        
                        if isinstance(arguments, dict):
                            query = arguments.get('search_query', '')
                            if query:
                                search_queries.append(query)
                
                if search_queries:
                    self.discussion_summary_manager.add_search_queries(
                        agent_id=self.agent_id,
                        search_queries=search_queries
                    )
                    self.logger.debug(f"{logger_name} 已保存 {len(search_queries)} 个搜索查询到讨论记录")
                else:
                    self.logger.debug(f"{logger_name} 未找到有效的搜索查询")

            
            self.logger.info(f"✅ {logger_name} 大模型生成完成")
            self.logger.debug(f"  响应长度: {len(response_text)} 字符")
            self.logger.debug(f"  工具调用次数: {len(tool_call_records)}")
            
            # ========== 4. 解析 XML 输出 ==========
            self.logger.debug(f"{logger_name} 解析生成结果...")
            
            from ..utils.helpers import parse_generation_result
            parsed_result = parse_generation_result(response_text)
            
            thinking = parsed_result.get('thinking', {})
            content = parsed_result.get('content', {})
            knowledge_points = parsed_result.get('knowledge_points', [])
            
            self.logger.info(f"✅ {logger_name} 成功解析生成结果")
            self.logger.debug(f"  思考过程: {len(thinking)} 个维度")
            self.logger.debug(f"  生成内容: {len(content)} 个部分")
            self.logger.debug(f"  知识点数量: {len(knowledge_points)}")
            
            # ========== 5. 将生成的知识点整合为完整报告存储到知识库 ==========
            stored_count = 0
            if self.knowledge_manager and knowledge_points:
                self.logger.debug(f"{logger_name} 开始整合 {len(knowledge_points)} 个知识点为完整报告...")
                
                # 构建完整的生成内容报告
                report_parts = [f"【生成内容报告 - {self.agent_id}】\n"]
                report_parts.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_parts.append(f"MBTI类型: {self.mbti_type}")
                report_parts.append(f"轮次: 第 {round_index}/{max_rounds} 轮" if round_index and max_rounds else "轮次: 未知")
                report_parts.append(f"知识点数量: {len(knowledge_points)} 个\n")
                
                # 逐个添加知识点到报告中
                for idx, kp in enumerate(knowledge_points, 1):
                    kp_title = kp.get('title', '未命名')
                    kp_content = kp.get('content', '')
                    kp_type = kp.get('type', 'unknown')
                    kp_importance = kp.get('importance', 'medium')
                    kp_source = kp.get('source', '未知')
                    
                    self.logger.debug(f"  整合知识点 {idx}/{len(knowledge_points)}: {kp_title}")
                    
                    # 添加到报告中，包含完整的元信息
                    report_parts.append(f"\n## {idx}. {kp_title}")
                    report_parts.append(f"类型: {kp_type} | 重要性: {kp_importance} | 来源: {kp_source}")
                    report_parts.append(kp_content)
                
                # 组合成完整报告
                integrated_report = "\n".join(report_parts)
                
                # 构建知识数据（整合后的单条记录）
                from src.utils.helpers import build_knowledge_with_meta
                knowledge_data = build_knowledge_with_meta(
                    knowledge_content=integrated_report,
                    agent_id=self.agent_id,
                    mbti_type=self.mbti_type,
                    task_id=self.current_task_id,
                    knowledge_type="generation_content_report",
                    importance="high",  # 整合报告默认高重要性
                    additional_meta={
                        'report_type': 'generation_content',
                        'knowledge_count': len(knowledge_points),
                        'round_index': round_index if round_index is not None else -1,
                        'integrated': True
                    }
                )
                
                try:
                    self.logger.info(f"📦 存储整合的生成内容报告（{len(knowledge_points)} 个知识点）...")
                    from ..tools.system.knowledgemanager import add_knowledge_batch_tool
                    result = add_knowledge_batch_tool(knowledge_list=[{
                        'knowledge': knowledge_data['content'],
                        'metadata': knowledge_data['metadata']
                    }])
                    result_dict = json.loads(result) if isinstance(result, str) else result
                    
                    if result_dict.get('success'):
                        stored_count = 1
                        self.logger.info(f"  ✓ 整合报告存储成功: 1 条记录")
                    else:
                        self.logger.error(f"  ✗ 整合报告存储失败: {result_dict.get('message')}")
                except Exception as e:
                    self.logger.error(f"  ✗ 整合报告存储异常: {e}")
                
                self.logger.info(f"✅ {logger_name} 生成完成，成功存储 {stored_count} 个知识点")
            
            # ========== 6. 返回结构化结果 ==========
            # 提取主要方案内容，优先使用解析后的 main_solution
            # 添加类型检查，防止 content 是字符串时报错
            if isinstance(content, dict):
                main_solution_text = content.get('main_solution', '')
            else:
                self.logger.warning(f"{logger_name} ⚠️ content 不是字典类型，使用原始响应")
                main_solution_text = response_text
                content = {}  # 重置为空字典避免后续错误
            
            # 如果解析后的 main_solution 为空，使用原始响应作为兜底
            if not main_solution_text:
                self.logger.warning(f"{logger_name} ⚠️ XML 解析未得到 main_solution，使用原始响应")
                main_solution_text = response_text
            
            return {
                'success': True,
                'main_solution': main_solution_text,
                'thinking_summary': thinking.get('context_understanding', '')[:200],  # 摘要
                'team_collaboration': content.get('team_collaboration', ''),
                'stakeholder_impact': content.get('stakeholder_impact', ''),
                'knowledge_points_count': stored_count,
                'web_search_count': web_search_count,
                'web_search_knowledge_count': web_search_knowledge_count,
                'full_thinking': thinking,
                'full_content': content,
                'raw_response': response_text,  # 添加原始响应用于兜底
                'error_message': ''
            }
            
        except FileNotFoundError as e:
            error_msg = f"提示词文件未找到: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'main_solution': f"【生成失败】{error_msg}",
                'thinking_summary': '',
                'team_collaboration': '',
                'knowledge_points_count': 0,
                'web_search_count': 0,
                'web_search_knowledge_count': 0,
                'error_message': error_msg
            }
        except Exception as e:
            error_msg = f"生成方案时发生异常: {e}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            return {
                'success': False,
                'main_solution': f"【生成失败】{error_msg}",
                'thinking_summary': '',
                'team_collaboration': '',
                'knowledge_points_count': 0,
                'web_search_count': 0,
                'web_search_knowledge_count': 0,
                'error_message': error_msg
            }


    def communicate(self,user_prompt)->str:
        """
        智能体沟通说话逻辑。只基于当前轮次进行对话。
        Args:
            user_prompt: 用户提示词
        Returns:
            回复内容
        
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        self.logger.info(f"💬 {self.agent_id} 开始沟通对话...")
        
        try:
            system_prompt = self.get_personality_prompt()
            self.logger.debug("构建沟通对话请求...")
            
            response_text = call_llm(
                llm_client=self.llm_client,
                model_name=self.config.model.model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                logger=self.logger,
                logger_name=logger_name
            )
            return response_text
            
        except Exception as e:
            self.logger.error(f"❌ 沟通对话时出错: {e}", exc_info=True)
            return "【沟通对话失败】"

    def attitude(self, task: str = None, message: Message = None, 
                 round_index=None, max_rounds=None) -> Dict[str, Any]:
        """
        智能体态度分析 - 在听取其他智能体发言时的思考动作。基于attitude函数来实现EVA机制。
        
        分析内容：
        1. 讨论完善度评估：当前方案是否完善、关键问题、共识状态
        2. 结束讨论建议：是否建议提前结束、理由、确定程度
        3. 发言意愿评估：是否想发言、意愿强度（程度词→分数）、理由
        4. 发言思路规划：核心观点、关键论点、强调方面（用于后续检索知识库）
        
        执行流程：
        1. 调用 LLMToolClient 获取 XML 格式的态度分析结果
        2. 解析 XML 提取各部分信息
        3. 将有价值的内容（不含发言思路）添加 meta 信息后存入知识库
        4. 更新发言意愿分数（用于发言队列排序）
        5. 更新发言思路上下文（用于 generate 函数检索）
        
        Args:
            task: 任务描述
            message: 最新收到的消息
            round_index: 当前讨论轮次
            max_rounds: 最大讨论轮次
            
        Returns:
            包含态度分析结果的字典：
            {
                'success': bool,  # 分析是否成功
                'discussion_evaluation': dict,  # 讨论评估
                'termination_suggestion': dict,  # 结束建议
                'speaking_intention': dict,  # 发言意愿
                'speaking_plan': dict,  # 发言计划
                'stored_to_kb': bool  # 是否成功存入知识库
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        self.logger.info(f"🤔 {logger_name} 开始态度分析...")
        
        # 检查必要组件
        if not self.llm_tool_client:
            self.logger.error(f"❌ {logger_name} LLMToolClient 未初始化")
            return {
                'success': False,
                'discussion_evaluation': {},
                'termination_suggestion': {},
                'speaking_intention': {},
                'representative_intention': {},
                'speaking_plan': {},
                'stored_to_kb': False
            }
        
        try:
            conversation_history = "[无历史消息]"
            # 1. 获取历史对话上下文
            if self.system_config.attitude_memory_window>0:
                self.logger.debug(f"{logger_name} 检索历史消息...")
                recent_messages = self._recall_memory(
                    top_k=self.system_config.attitude_memory_window,
                    stages=stages
                )
                
                
                if len(recent_messages) > 1:
                    self.logger.debug(f"{logger_name} 构建对话历史（{len(recent_messages)-1} 条消息）...")
                    history_messages = recent_messages[1:]  # 第一条是最新消息，不纳入历史
                    conversation_history = self._format_conversation_history(history_messages)
                
            # 2. 构建轮次提示
            round_prompt = ""
            if round_index is not None and max_rounds is not None:
                round_prompt = f"【当前讨论轮次】\n这是第 {round_index} 轮讨论，总共 {max_rounds} 轮。需要尽可能在最大轮次之前与队友达成一致方案。\n"
                self.logger.debug(f"{logger_name} 当前轮次: {round_index}/{max_rounds}")
            
            # 3. 加载提示词模板
            self.logger.debug(f"{logger_name} 加载提示词模板...")
            user_prompt_template = prompt_loader.load_function(
                mbti_type=self.mbti_type.lower(),
                function_name=task_name
            )
            
            # 4. 格式化提示词
            user_prompt = user_prompt_template.format(
                agent_id=self.agent_id,
                task=task or "未指定任务",
                round_prompt=round_prompt,
                sender_id=message.sender_id if message else "未知",
                message_content=message.content if message else "无内容",
                conversation_history=conversation_history
            )
            
            self.logger.debug(f"{logger_name} 提示词准备完成")
            self.logger.debug(f"  用户提示词长度: {len(user_prompt)} 字符, 内容:\n{user_prompt}")
            
            # 5. 调用 LLM（不使用工具调用）
            self.logger.info(f"🤖 {logger_name} 调用大模型进行态度分析...")
            
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=self.config.model.model_name,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens
            )
            self.logger.debug(f"  LLM Response: {response}")
        
            # 提取响应文本
            response_text = response.get('content', '')
            
            if not response_text:
                self.logger.error(f"❌ {logger_name} 大模型返回空响应")
                return {
                    'success': False,
                    'discussion_evaluation': {},
                    'termination_suggestion': {},
                    'speaking_intention': {},
                    'representative_intention': {},
                    'speaking_plan': {},
                    'stored_to_kb': False
                }
            
            self.logger.info(f"✓ {logger_name} 大模型分析完成")
            self.logger.debug(f"  响应长度: {len(response_text)} 字符")
            
            # 6. 解析 XML 提取态度信息
            self.logger.debug(f"{logger_name} 解析态度分析结果...")
            from src.utils.helpers import parse_attitude_result, build_knowledge_with_meta
            
            parsed_result = parse_attitude_result(response_text)
            
            if not parsed_result.get('discussion_evaluation'):
                self.logger.warning(f"⚠️ {logger_name} 未能解析出有效的态度分析结果")
                return {
                    'success': False,
                    'discussion_evaluation': {},
                    'termination_suggestion': {},
                    'speaking_intention': {},
                    'representative_intention': {},
                    'speaking_plan': {},
                    'stored_to_kb': False
                }
            
            self.logger.info(f"✓ {logger_name} 态度分析结果解析成功")
            self.logger.debug(f"  讨论完善度: {parsed_result['discussion_evaluation'].get('completeness_degree', '未知')}")
            self.logger.debug(f"  建议结束: {parsed_result['termination_suggestion'].get('should_terminate', False)}")
            self.logger.debug(f"  发言意愿: {parsed_result['speaking_intention'].get('intention_degree', '未知')} ({parsed_result['speaking_intention'].get('intention_score', 0)}分)")
            self.logger.debug(f"  代表意愿: {parsed_result['representative_intention'].get('representative_degree', '未知')} ({parsed_result['representative_intention'].get('representative_score', 0)}分)")
            
            # 7. 将态度分析结果整合为完整报告存入知识库
            stored_to_kb = False
            if self.knowledge_manager and parsed_result:
                try:
                    self.logger.debug(f"{logger_name} 开始整合态度分析为完整报告...")
                    
                    # 构建完整的态度分析报告
                    report_parts = [f"【态度分析报告 - {self.agent_id}】\n"]
                    report_parts.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    report_parts.append(f"MBTI类型: {self.mbti_type}")
                    report_parts.append(f"轮次: {round_index}/{max_rounds}" if round_index and max_rounds else "轮次: 未知")
                    report_parts.append(f"消息发送者: {message.sender_id if message else '无'}\n")
                    
                    # 添加讨论完善度评估
                    discussion_eval = parsed_result.get('discussion_evaluation', {})
                    if discussion_eval:
                        report_parts.append("## 讨论完善度评估")
                        report_parts.append(f"完善程度: {discussion_eval.get('completeness_degree', '未知')}")
                        report_parts.append(f"分析: {discussion_eval.get('completeness_analysis', '无')}")
                        report_parts.append(f"关键问题: {discussion_eval.get('key_issues', '无')}")
                        report_parts.append(f"共识状态: {discussion_eval.get('consensus_status', '未知')}\n")
                    
                    # 添加结束讨论建议
                    termination = parsed_result.get('termination_suggestion', {})
                    if termination:
                        report_parts.append("## 结束讨论建议")
                        report_parts.append(f"建议结束: {'是' if termination.get('should_terminate') else '否'}")
                        report_parts.append(f"理由: {termination.get('termination_reason', '无')}")
                        report_parts.append(f"确定程度: {termination.get('confidence_level', '未知')}\n")
                    
                    # 添加发言意愿评估
                    speaking_intention = parsed_result.get('speaking_intention', {})
                    if speaking_intention:
                        report_parts.append("## 发言意愿评估")
                        report_parts.append(f"是否想发言: {'是' if speaking_intention.get('desire_to_speak') else '否'}")
                        report_parts.append(f"意愿程度: {speaking_intention.get('intention_degree', '未知')} (分数: {speaking_intention.get('intention_score', 0)})")
                        report_parts.append(f"理由: {speaking_intention.get('intention_reason', '无')}\n")
                    
                    # 添加代表发言意愿评估
                    representative = parsed_result.get('representative_intention', {})
                    if representative:
                        report_parts.append("## 代表发言意愿")
                        report_parts.append(f"是否愿意担任代表: {'是' if representative.get('desire_to_represent') else '否'}")
                        report_parts.append(f"意愿程度: {representative.get('representative_degree', '未知')} (分数: {representative.get('representative_score', 0)})")
                        report_parts.append(f"理由: {representative.get('representative_reason', '无')}\n")
                    
                    # 添加发言思路规划（用于后续检索）
                    speaking_plan = parsed_result.get('speaking_plan', {})
                    if speaking_plan:
                        report_parts.append("## 发言思路规划")
                        report_parts.append(f"核心观点: {speaking_plan.get('core_viewpoint', '无')}")
                        report_parts.append(f"关键论点: {speaking_plan.get('key_points', '无')}")
                        report_parts.append(f"强调方面: {speaking_plan.get('emphasis_aspects', '无')}")
                    
                    # 组合成完整报告
                    integrated_report = "\n".join(report_parts)
                    
                    # 构建知识数据（整合后的单条记录）
                    knowledge_entry = build_knowledge_with_meta(
                        knowledge_content=integrated_report,
                        agent_id=self.agent_id,
                        mbti_type=self.mbti_type,
                        task_id=self.current_task_id,
                        knowledge_type='attitude_analysis_report',
                        importance='high',  # 整合报告默认高重要性
                        additional_meta={
                            'report_type': 'attitude_analysis',
                            'round_index': round_index,
                            'sender_id': message.sender_id if message else None,
                            'intention_score': speaking_intention.get('intention_score', 0),
                            'should_terminate': termination.get('should_terminate', False),
                            'integrated': True
                        }
                    )
                    
                    # 存储整合后的单条报告
                    from ..tools.system.knowledgemanager import add_knowledge_batch_tool
                    result_json = add_knowledge_batch_tool(knowledge_list=[{
                        'knowledge': knowledge_entry['content'],
                        'metadata': knowledge_entry['metadata']
                    }])
                    result_dict = json.loads(result_json) if isinstance(result_json, str) else result_json
                    
                    if result_dict.get('success'):
                        stored_to_kb = True
                        self.logger.info(f"✓ {logger_name} 整合的态度分析报告已存入知识库")
                    else:
                        self.logger.warning(f"⚠️ {logger_name} 知识库存储失败: {result_dict.get('message')}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {logger_name} 存入知识库时出错: {e}", exc_info=True)
            else:
                if not self.knowledge_manager:
                    self.logger.debug(f"{logger_name} 知识库管理器未初始化，跳过存储")
                else:
                    self.logger.debug(f"{logger_name} 无有效分析结果，跳过存储")
            
            # 8. 更新发言意愿（用于发言队列）
            if parsed_result.get('speaking_intention'):
                intention_score = parsed_result['speaking_intention'].get('intention_score', 6.0)
                intention_reason = parsed_result['speaking_intention'].get('intention_reason', '无具体理由')
                
                self.update_speaking_intention(intention_score, intention_reason)
                self.logger.info(f"✓ {logger_name} 发言意愿已更新: {intention_score}分")
            
            # 9. 更新发言思路上下文（用于 generate 函数检索）
            if parsed_result.get('speaking_plan'):
                plan = parsed_result['speaking_plan']
                self.speaking_plan_context = f"{plan.get('core_viewpoint', '')}\n{plan.get('key_points', '')}\n{plan.get('emphasis_aspects', '')}"
                self.logger.info(f"✓ {logger_name} 发言思路上下文已更新")
                self.logger.debug(f"  核心观点: {plan.get('core_viewpoint', '无')}")
            
            # 10. 保存到讨论摘要
            if self.discussion_summary_manager and parsed_result.get('speaking_plan'):
                plan = parsed_result['speaking_plan']
                core_viewpoint = plan.get('core_viewpoint', '')
                if core_viewpoint:
                    self.discussion_summary_manager.add_summary(
                        action="attitude",
                        agent_id=self.agent_id,
                        content=core_viewpoint[:200],
                        metadata={
                            "message_sender": message.sender_id if message else None
                        }
                    )
                    self.logger.debug(f"{logger_name} 已保存态度摘要到讨论记录")
            
            # 11. 返回完整结果
            return {
                'success': True,
                'discussion_evaluation': parsed_result.get('discussion_evaluation', {}),
                'termination_suggestion': parsed_result.get('termination_suggestion', {}),
                'speaking_intention': parsed_result.get('speaking_intention', {}),
                'representative_intention': parsed_result.get('representative_intention', {}),
                'speaking_plan': parsed_result.get('speaking_plan', {}),
                'stored_to_kb': stored_to_kb
            }
            
        except Exception as e:
            self.logger.error(f"❌ {logger_name} 态度分析时出错: {e}", exc_info=True)
            return {
                'success': False,
                'discussion_evaluation': {},
                'termination_suggestion': {},
                'speaking_intention': {},
                'representative_intention': {},
                'speaking_plan': {},
                'stored_to_kb': False,
                'error': str(e)
            }

    def collate(self, task: str, min_words: int = 6000) -> Dict[str, Any]:
        """
        整理信息生成完整文章 - 基于知识库的结构化生成
        
        执行流程：
        1. 从知识库检索相关知识
        2. 生成撰写大纲（XML格式）
        3. 解析大纲提取章节信息
        4. 基于章节关键语句检索知识库
        5. 逐章节生成内容并存入知识库
        6. 组装完整文章（XML格式）
        7. 解析出纯文本版本
        
        Args:
            task: 任务描述
            min_words: 期望的最小字数（默认6000）
            
        Returns:
            Dict[str, Any]: 包含生成结果的字典
            {
                'success': bool,  # 生成是否成功
                'article_html': str,  # HTML格式的完整文章
                'article_text': str,  # 纯文本版本（无XML标签）
                'outline': Dict,  # 文章大纲
                'sections_count': int,  # 章节数量
                'total_words': int,  # 总字数
                'knowledge_points_stored': int,  # 存入知识库的知识点数量
                'references_count': int,  # 参考文献数量
                'timestamp': str,  # 生成时间戳
                'error_message': str  # 错误信息
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        
        self.logger.info(f"📝 {logger_name} 开始整理文章")
        self.logger.info(f"  任务: {task[:50]}...")
        self.logger.info(f"  目标字数: {min_words}")
        
        # 检查必要组件
        if not self.llm_tool_client:
            error_msg = "LLMToolClient 未初始化"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            return self._collate_error_result(error_msg)
        
        if not self.knowledge_manager:
            error_msg = "知识库管理器未初始化"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            return self._collate_error_result(error_msg)
        
        try:
            # ========== 步骤0: 获取讨论摘要 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"💬 步骤0: 获取讨论摘要")
            self.logger.info(f"{'='*60}")
            
            discussion_summary = ""
            if self.discussion_summary_manager:
                discussion_summary = self.discussion_summary_manager.get_formatted_summary(max_items=50)
                stats = self.discussion_summary_manager.get_summary_statistics()
                self.logger.info(f"✓ 获取讨论摘要: {len(discussion_summary)} 字符")
                self.logger.info(f"  统计信息: {stats}")
            else:
                self.logger.warning("⚠️ 讨论摘要管理器未初始化")
            
            # ========== 步骤1: 生成撰写大纲（不再使用知识库检索）==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📋 步骤1: 生成撰写大纲")
            self.logger.info(f"{'='*60}")
            
            outline_result = self._generate_collate_outline(task, discussion_summary, min_words)
            
            if not outline_result['success']:
                error_msg = f"大纲生成失败: {outline_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                return self._collate_error_result(error_msg)
            
            outline_sections = outline_result['sections']
            self.logger.info(f"✓ 大纲生成完成，共 {len(outline_sections)} 个章节")
            
            # ========== 步骤2: 逐章节生成内容 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📄 步骤2: 逐章节生成内容")
            self.logger.info(f"{'='*60}")
            
            sections_data = []
            covered_sections = []
            
            for idx, section_info in enumerate(outline_sections, 1):
                section_title = section_info.get('title', f'章节{idx}')
                section_focus = section_info.get('focus', '')
                key_phrases = section_info.get('key_phrases', [])
                word_target = section_info.get('word_target', min_words // len(outline_sections))
                
                self.logger.info(f"\n📝 生成第 {idx}/{len(outline_sections)} 章: {section_title}")
                self.logger.debug(f"  撰写重点: {section_focus}")
                self.logger.debug(f"  目标字数: {word_target}")
                
                # 基于关键语句检索知识section_title、section_focus、key_phrases三合一
                search_phrases = [section_title, section_focus] + key_phrases
                self.logger.debug(f"  检索关键语句: {search_phrases}")
                section_knowledge = self._retrieve_knowledge_for_section(search_phrases,top_k_per_phrase=4) # TODO top_k_per_phrase 放到配置文件中
                self.logger.debug(f"  检索到 {len(section_knowledge)} 条章节相关知识")
                
                # 生成章节内容
                section_result = self._generate_section_content(
                    task=task,
                    section_index=idx,
                    total_sections=len(outline_sections),
                    section_title=section_title,
                    section_focus=section_focus,
                    section_knowledge=section_knowledge,
                    word_target=word_target,
                    covered_sections=covered_sections
                )
                
                if not section_result['success']:
                    self.logger.warning(f"⚠️ 第 {idx} 章生成失败: {section_result['error_message']}")
                    # 继续生成下一章节
                    continue
                
                # 存储章节数据
                sections_data.append({
                    'title': section_title,
                    'content': section_result['content'],
                    'html': section_result.get('html', ''),  # 添加HTML字段
                    'references': section_result['references']
                })
                
                # 记录已覆盖章节
                covered_sections.append(section_title)
                
                self.logger.info(f"✓ 第 {idx} 章完成 ({len(section_result['content'])} 字)")
            
            # ========== 步骤4: 组装完整文章 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📦 步骤4: 组装完整文章")
            self.logger.info(f"{'='*60}")
            
            article_result = self._assemble_article(task, sections_data)
            
            if not article_result['success']:
                error_msg = f"文章组装失败: {article_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                return self._collate_error_result(error_msg)
            
            # ========== 构建返回结果 ==========
            result = {
                'success': True,
                'article_html': article_result['article_html'],
                'article_text': article_result['article_text'],
                'outline': {'sections': outline_sections},
                'sections_count': len(sections_data),
                'total_words': len(article_result['article_text']),
                'knowledge_points_stored': 0,
                'references_count': article_result['references_count'],
                'timestamp': datetime.now().isoformat(),
                'error_message': ''
            }
            
            self.logger.info(f"✅ {logger_name} 文章生成完成")
            self.logger.info(f"  章节数: {result['sections_count']}")
            self.logger.info(f"  总字数: {result['total_words']}")
            self.logger.info(f"  参考文献: {result['references_count']}")
            
            return result
            
        except Exception as e:
            error_msg = f"整理文章时发生异常: {str(e)}"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            import traceback
            self.logger.debug(f"异常堆栈:\n{traceback.format_exc()}")
            return self._collate_error_result(error_msg)
    
    def _collate_error_result(self, error_message: str) -> Dict[str, Any]:
        """构建 collate 的错误返回结果"""
        return {
            'success': False,
            'article_html': '',
            'article_text': '',
            'outline': {},
            'sections_count': 0,
            'total_words': 0,
            'knowledge_points_stored': 0,
            'references_count': 0,
            'timestamp': datetime.now().isoformat(),
            'error_message': error_message
        }
    
    def collate_single(self, task: str, min_words: int = 6000) -> Dict[str, Any]:
        """
        一次性生成完整文章 - 使用强大模型直接生成，无需大纲分章节
        
        执行流程：
        1. 从知识库检索相关知识
        2. 将知识分为网页知识和讨论知识两类
        3. 获取讨论摘要
        4. 使用 collate_single.txt 提示词调用专用大模型
        5. 解析生成的 HTML 格式文章
        6. 提取纯文本版本
        
        Args:
            task: 任务描述
            min_words: 期望的最小字数（默认6000）
            
        Returns:
            Dict[str, Any]: 包含生成结果的字典
            {
                'success': bool,  # 生成是否成功
                'article_html': str,  # HTML格式的完整文章
                'article_text': str,  # 纯文本版本（无XML标签）
                'total_words': int,  # 总字数
                'references_count': int,  # 参考文献数量
                'timestamp': str,  # 生成时间戳
                'error_message': str  # 错误信息
            }
        """
        task_name = inspect.currentframe().f_code.co_name
        logger_name = f"【{self.agent_id};{task_name}】"
        
        self.logger.info(f"📝 {logger_name} 开始一次性生成文章")
        self.logger.info(f"  任务: {task[:50]}...")
        self.logger.info(f"  目标字数: {min_words}")
        
        # 检查必要组件
        if not self.llm_tool_client:
            error_msg = "LLMToolClient 未初始化"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            return self._collate_error_result(error_msg)
        
        if not self.knowledge_manager:
            error_msg = "知识库管理器未初始化"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            return self._collate_error_result(error_msg)
        
        try:
            # ========== 步骤1: 检索知识库 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔍 步骤1: 检索知识库")
            self.logger.info(f"{'='*60}")
            
            knowledge_list = self._retrieve_knowledge_for_collate(task, top_k=20)
            self.logger.info(f"✓ 检索到 {len(knowledge_list)} 条知识")
            
            # ========== 步骤2: 分类知识（网页知识 vs 讨论知识）==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📊 步骤2: 分类知识")
            self.logger.info(f"{'='*60}")
            
            web_knowledge = []  # 网页知识（有URL）
            discussion_knowledge = []  # 讨论知识（智能体讨论）
            
            for item in knowledge_list:
                metadata = item.get('metadata', {})
                source_type = metadata.get('source_type', 'unknown')
                url = metadata.get('url') or metadata.get('source_url', '')
                
                # 判断是否为网页知识
                if url and url.strip() and not url.startswith('unknown'):
                    web_knowledge.append(item)
                else:
                    discussion_knowledge.append(item)
            
            self.logger.info(f"✓ 网页知识: {len(web_knowledge)} 条")
            self.logger.info(f"✓ 讨论知识: {len(discussion_knowledge)} 条")
            
            # ========== 步骤3: 获取讨论摘要 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"💬 步骤3: 获取讨论摘要")
            self.logger.info(f"{'='*60}")
            
            discussion_summary = ""
            if self.discussion_summary_manager:
                discussion_summary = self.discussion_summary_manager.get_formatted_summary(max_items=50)
                stats = self.discussion_summary_manager.get_summary_statistics()
                self.logger.info(f"✓ 获取讨论摘要: {len(discussion_summary)} 字符")
                self.logger.info(f"  统计信息: {stats}")
            else:
                self.logger.warning("⚠️ 讨论摘要管理器未初始化")
            
            # ========== 步骤4: 格式化知识为提示词 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📝 步骤4: 格式化知识")
            self.logger.info(f"{'='*60}")
            
            # 格式化网页知识（带引用编号）
            web_knowledge_text_parts = []
            ref_counter = 1
            ref_url_mapping = {}  # {ref_id: url}
            
            if web_knowledge:
                web_knowledge_text_parts.append("### 网页知识（可引用）\n")
                for item in web_knowledge:
                    content = item.get('content', '')
                    metadata = item.get('metadata', {})
                    url = metadata.get('url') or metadata.get('source_url', '')
                    source_description = metadata.get('source_description', '网页来源')
                    
                    ref_id = str(ref_counter)
                    ref_url_mapping[ref_id] = {
                        'url': url,
                        'description': source_description
                    }
                    
                    web_knowledge_text_parts.append(
                        f"\n[网页知识 - 可引用编号 {ref_counter}]\n"
                        f"内容: {content}\n"
                        f"来源: {source_description}\n"
                        f"URL: {url}\n"
                    )
                    ref_counter += 1
            
            # 格式化讨论知识（不可引用）
            discussion_knowledge_text_parts = []
            if discussion_knowledge:
                discussion_knowledge_text_parts.append("\n### 讨论知识（不可引用，仅供参考）\n")
                for item in discussion_knowledge:
                    content = item.get('content', '')
                    metadata = item.get('metadata', {})
                    agent_id = metadata.get('agent_id', '未知智能体')
                    
                    discussion_knowledge_text_parts.append(
                        f"\n[讨论知识]\n"
                        f"内容: {content}\n"
                        f"来源: 智能体讨论 ({agent_id})\n"
                    )
            
            # 合并知识文本
            formatted_knowledge = ''.join(web_knowledge_text_parts + discussion_knowledge_text_parts)
            
            if not formatted_knowledge.strip():
                formatted_knowledge = "（暂无相关知识）"
            
            self.logger.info(f"✓ 知识格式化完成，总长度: {len(formatted_knowledge)} 字符")
            
            # ========== 步骤5: 加载提示词并生成文章 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🤖 步骤5: 调用大模型生成文章")
            self.logger.info(f"{'='*60}")
            
            # 加载提示词模板
            prompt_template = prompt_loader.load_system_prompt('collate_single')
            user_prompt = prompt_template.format(
                task=task,
                min_words=min_words,
                discussion_summary=discussion_summary if discussion_summary else "（暂无讨论摘要）",
                formatted_knowledge=formatted_knowledge
            )
            
            self.logger.debug(f"提示词准备完成，长度: {len(user_prompt)} 字符")
            
            # 构建消息列表
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            # 获取 collate 专用配置
            if hasattr(self, 'collate_llm_client') and self.collate_llm_client and self.system_config and hasattr(self.system_config, 'collate_model') and self.system_config.collate_model:
                model_name = self.system_config.collate_model.model_name
                max_tokens = self.system_config.collate_model.max_tokens
                temperature = self.system_config.collate_model.temperature
                self.logger.info(f"  使用 Collate 专用模型: {model_name}")
            else:
                model_name = self.config.model.model_name
                max_tokens = self.config.model.max_tokens
                temperature = 0.8
                self.logger.info(f"  使用默认模型: {model_name}")
            
            # 调用 LLM
            self.logger.info(f"  调用参数: temperature={temperature}, max_tokens={max_tokens}")
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取响应文本
            response_text = response.get('content', '')
            self.logger.info(f"✓ 大模型响应完成，长度: {len(response_text)} 字符")
            self.logger.debug(f"响应内容: {response_text[:500]}...")
            
            # ========== 步骤6: 解析 HTML 文章 ==========
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔧 步骤6: 解析 HTML 文章")
            self.logger.info(f"{'='*60}")
            
            parsed_result = self._parse_single_article_html(response_text, ref_url_mapping)
            
            if not parsed_result['success']:
                error_msg = f"文章解析失败: {parsed_result['error_message']}"
                self.logger.error(f"❌ {error_msg}")
                return self._collate_error_result(error_msg)
            
            # ========== 构建返回结果 ==========
            result = {
                'success': True,
                'article_html': parsed_result['article_html'],
                'article_text': parsed_result['article_text'],
                'total_words': len(parsed_result['article_text']),
                'references_count': parsed_result['references_count'],
                'timestamp': datetime.now().isoformat(),
                'error_message': ''
            }
            
            self.logger.info(f"✅ {logger_name} 文章生成完成")
            self.logger.info(f"  总字数: {result['total_words']}")
            self.logger.info(f"  参考文献: {result['references_count']}")
            
            return result
            
        except Exception as e:
            error_msg = f"生成文章时发生异常: {str(e)}"
            self.logger.error(f"❌ {logger_name} {error_msg}")
            import traceback
            self.logger.debug(f"异常堆栈:\n{traceback.format_exc()}")
            return self._collate_error_result(error_msg)
    
    def _parse_single_article_html(self, html_text: str, ref_url_mapping: Dict[str, Dict]) -> Dict[str, Any]:
        """
        解析一次性生成的完整文章HTML
        
        Args:
            html_text: HTML格式的文章文本
            ref_url_mapping: 引用编号到URL的映射 {ref_id: {'url': ..., 'description': ...}}
            
        Returns:
            Dict包含:
            - success: bool
            - article_html: str（保留完整HTML）
            - article_text: str（纯文本版本）
            - references_count: int
            - error_message: str
        """
        try:
            from bs4 import BeautifulSoup
            
            # 清理HTML文本（移除可能的markdown代码块标记）
            html_text = html_text.strip()
            if html_text.startswith('```html'):
                html_text = html_text[7:]
            if html_text.endswith('```'):
                html_text = html_text[:-3]
            html_text = html_text.strip()
            
            # 解析HTML
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # 查找article标签
            article = soup.find('article', {'data-report': 'true'})
            if not article:
                # 尝试查找任何article标签
                article = soup.find('article')
                if not article:
                    return {
                        'success': False,
                        'article_html': '',
                        'article_text': '',
                        'references_count': 0,
                        'error_message': '未找到<article>标签'
                    }
            
            # 提取标题
            title_tag = article.find('h1', {'data-field': 'title'})
            title = title_tag.get_text(strip=True) if title_tag else "研究报告"
            
            # 提取主体内容
            main = article.find('main')
            if not main:
                return {
                    'success': False,
                    'article_html': '',
                    'article_text': '',
                    'references_count': 0,
                    'error_message': '未找到<main>标签'
                }
            
            # 构建纯文本版本
            text_parts = []
            text_parts.append(f"# {title}\n\n")
            
            # 遍历所有section
            for section in main.find_all('section', recursive=False):
                # 提取章节标题
                h2 = section.find('h2')
                if h2:
                    section_title = h2.get_text(strip=True)
                    text_parts.append(f"## {section_title}\n\n")
                
                # 提取段落和子标题
                for element in section.find_all(['h3', 'p']):
                    if element.name == 'h3':
                        text_parts.append(f"### {element.get_text(strip=True)}\n\n")
                    elif element.name == 'p':
                        # 转换引用标记为[n]格式
                        text = self._convert_html_citations_to_text(element)
                        text_parts.append(f"{text}\n\n")
            
            # 提取参考文献
            footer = article.find('footer')
            references_count = 0
            
            if footer:
                refs_ol = footer.find('ol', class_='references')
                if refs_ol:
                    ref_items = refs_ol.find_all('li', attrs={'data-ref-id': True})
                    references_count = len(ref_items)
                    
                    if references_count > 0:
                        text_parts.append("\n## 参考文献\n\n")
                        
                        # 提取所有引用并去重（按URL）
                        url_to_refs = {}
                        for ref_li in ref_items:
                            ref_id = ref_li.get('data-ref-id', '')
                            cite = ref_li.find('cite')
                            if cite:
                                url_span = cite.find('span', {'data-field': 'url'})
                                desc_span = cite.find('span', {'data-field': 'description'})
                                
                                if url_span:
                                    url = url_span.get_text(strip=True)
                                    desc = desc_span.get_text(strip=True) if desc_span else '来源'
                                    
                                    if url not in url_to_refs:
                                        url_to_refs[url] = {
                                            'description': desc,
                                            'ids': []
                                        }
                                    url_to_refs[url]['ids'].append(ref_id)
                        
                        # 构建去重后的参考文献列表
                        counter = 1
                        for url, info in url_to_refs.items():
                            desc = info['description']
                            text_parts.append(f"[{counter}] {desc}. {url}\n")
                            counter += 1
            
            article_text = ''.join(text_parts)
            
            # 返回完整的HTML（保持原样）
            article_html = str(article)
            
            return {
                'success': True,
                'article_html': article_html,
                'article_text': article_text,
                'references_count': references_count,
                'error_message': ''
            }
            
        except Exception as e:
            return {
                'success': False,
                'article_html': '',
                'article_text': '',
                'references_count': 0,
                'error_message': str(e)
            } 


    
    def _retrieve_knowledge_for_generate(self, task: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """从知识库检索与生成任务相关的知识（用于 generate 函数）"""
        try:
            # 使用适中的阈值检索，平衡相关性和召回率
            result = query_knowledge_tool(
                query=task,
                top_k=top_k,
                distance_threshold=1.5  # 适中阈值
            )
            
            # 解析结果
            import json
            result_data = json.loads(result)
            
            if not result_data.get('success', False):
                self.logger.warning(f"⚠️ 知识检索失败: {result_data.get('message', '未知错误')}")
                return []
            
            knowledge_list = result_data.get('results', [])
            self.logger.debug(f"  ✓ 成功检索到 {len(knowledge_list)} 条知识")
            return knowledge_list
            
        except Exception as e:
            self.logger.error(f"❌ 检索知识时出错: {e}")
            return []
    
    def _retrieve_knowledge_for_collate(self, task: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """从知识库检索与任务相关的知识"""
        try:
            # 使用较宽泛的阈值检索
            result = query_knowledge_tool(
                query=task,
                top_k=top_k,
                distance_threshold=1.5  # 宽泛阈值
            )
            
            # 解析结果
            import json
            result_data = json.loads(result)
            
            if not result_data.get('success', False):
                self.logger.warning(f"⚠️ 知识检索失败: {result_data.get('message', '未知错误')}")
                return []
            
            knowledge_list = result_data.get('results', [])
            return knowledge_list
            
        except Exception as e:
            self.logger.error(f"❌ 检索知识时出错: {e}")
            return []
    
    def _generate_collate_outline(self, task: str, discussion_summary: str, 
                                  min_words: int) -> Dict[str, Any]:
        """生成文章大纲（基于讨论摘要）"""
        try:
            # 加载提示词模板
            prompt_template = prompt_loader.load_system_prompt('collate_outline')
            user_prompt = prompt_template.format(
                task=task,
                discussion_summary=discussion_summary,
                min_words=min_words
            )
            self.logger.debug(f"构建大纲生成提示词：\n{user_prompt}")
            # 构建消息列表
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            # 获取 collate 专用配置
            if hasattr(self, 'collate_llm_client') and self.collate_llm_client and self.system_config and hasattr(self.system_config, 'collate_model') and self.system_config.collate_model:
                model_name = self.system_config.collate_model.model_name
                max_tokens = self.system_config.collate_model.max_tokens
                temperature = self.system_config.collate_model.temperature
                self.logger.debug(f"  使用 Collate 专用模型: {model_name}")
            else:
                model_name = self.config.model.model_name
                max_tokens = self.config.model.max_tokens
                temperature = 0.7
                self.logger.debug(f"  使用默认模型: {model_name}")
            
            # 调用 LLM
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.logger.debug(f"  LLM Response: {response}")
        
            # 提取响应文本
            response_text = response.get('content', '')
            
            # 调试：打印响应文本
            self.logger.debug(f"大纲生成响应: {response_text}")
            
            # 解析 HTML 提取 <outline> 标签
            sections = self._parse_outline_html(response_text)
            
            if not sections:
                return {
                    'success': False,
                    'sections': [],
                    'error_message': '大纲解析失败，未找到有效章节'
                }
            
            return {
                'success': True,
                'sections': sections,
                'error_message': ''
            }
            
        except Exception as e:
            return {
                'success': False,
                'sections': [],
                'error_message': str(e)
            }
    
    def _parse_outline_html(self, xml_text: str) -> List[Dict[str, Any]]:
        """解析大纲 HTML（之前是XML，现已改为HTML格式）"""
        try:
            from bs4 import BeautifulSoup
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(xml_text, 'html.parser')
            
            # 查找article标签
            article = soup.find('article', {'data-outline': 'true'})
            if not article:
                # 兼容：如果没有data-outline属性，尝试直接查找article
                article = soup.find('article')
            
            if not article:
                self.logger.warning("⚠️ 未找到 <article> 标签")
                return []
            
            sections = []
            # 查找所有section标签
            for section in article.find_all('section'):
                # 提取章节索引和字数目标（从属性中）
                section_index = section.get('data-section-index', '0')
                word_target = section.get('data-word-target', '1000')
                
                # 提取标题
                title_elem = section.find('h2', {'data-field': 'title'})
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                # 提取聚焦点
                focus_elem = section.find('p', {'data-field': 'focus'})
                focus = focus_elem.get_text(strip=True) if focus_elem else ''
                
                # 提取关键语句列表
                key_phrases = []
                key_phrases_ul = section.find('ul', {'data-field': 'key-phrases'})
                if key_phrases_ul:
                    for li in key_phrases_ul.find_all('li'):
                        phrase_text = li.get_text(strip=True)
                        if phrase_text:
                            key_phrases.append(phrase_text)
                
                sections.append({
                    'index': int(section_index) if section_index else 0,
                    'title': title,
                    'focus': focus,
                    'key_phrases': key_phrases,
                    'word_target': int(word_target) if word_target else 1000
                })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"❌ 解析大纲 HTML 时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _retrieve_knowledge_for_section(self, key_phrases: List[str], 
                                       top_k_per_phrase: int = 4) -> List[Dict[str, Any]]:
        """基于关键语句检索章节相关知识"""
        all_knowledge = []
        seen_ids = set()
        
        for phrase in key_phrases:
            try:
                result = query_knowledge_tool(
                    query=phrase,
                    top_k=top_k_per_phrase,
                    distance_threshold=1.5  # 适中阈值
                )
                
                import json
                result_data = json.loads(result)
                
                if result_data.get('success', False):
                    for item in result_data.get('results', []):
                        # 去重
                        item_id = item.get('id', '')
                        if item_id and item_id not in seen_ids:
                            all_knowledge.append(item)
                            seen_ids.add(item_id)
                            
            except Exception as e:
                self.logger.warning(f"⚠️ 检索关键语句 '{phrase}' 时出错: {e}")
                continue
        return all_knowledge
    
    def _generate_section_content(self, task: str, section_index: int, 
                                  total_sections: int, section_title: str,
                                  section_focus: str, section_knowledge: List[Dict],
                                  word_target: int, covered_sections: List[str]) -> Dict[str, Any]:
        """生成章节内容"""
        try:
            # 格式化章节知识
            knowledge_text = self._format_knowledge_for_prompt(section_knowledge)
            
            # 收集有效 URL (白名单) - 兼容两种字段名
            # 同时构建 REF-n 到 URL 的映射
            valid_urls = set()
            ref_url_mapping = {}  # {"1": "https://...", "2": "https://..."}
            ref_counter = 1
            for item in section_knowledge:
                metadata = item.get('metadata', {})
                url = metadata.get('url') or metadata.get('source_url', '')
                if url and url.strip() and not url.startswith('unknown'):
                    valid_urls.add(url)
                    ref_url_mapping[str(ref_counter)] = url
                    ref_counter += 1
            
            # 格式化已覆盖章节
            covered_text = '\n'.join([f"- {title}" for title in covered_sections]) if covered_sections else "无（这是第一章）"
            
            # 加载提示词模板
            prompt_template = prompt_loader.load_system_prompt('collate_section')
            user_prompt = prompt_template.format(
                task=task,
                section_index=section_index,
                total_sections=total_sections,
                section_title=section_title,
                section_focus=section_focus,
                section_knowledge=knowledge_text,
                word_target=word_target,
                covered_sections=covered_text
            )
            self.logger.debug(f"章节{section_index}提示词准备完成:\n{user_prompt}")
            # 构建消息列表
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            # 获取 collate 专用配置
            if hasattr(self, 'collate_llm_client') and self.collate_llm_client and self.system_config and hasattr(self.system_config, 'collate_model') and self.system_config.collate_model:
                model_name = self.system_config.collate_model.model_name
                max_tokens = self.system_config.collate_model.max_tokens
                temperature = self.system_config.collate_model.temperature
                self.logger.debug(f"  章节{section_index}使用 Collate 专用模型: {model_name}")
            else:
                model_name = self.config.model.model_name
                max_tokens = self.config.model.max_tokens
                temperature = 0.8
                self.logger.debug(f"  章节{section_index}使用默认模型: {model_name}")
            
            # 调用 LLM
            response = self.llm_tool_client.call_without_tools(
                messages=messages,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.logger.debug(f"  LLM Response: {response}")
        
            # 提取响应文本
            response_text = response.get('content', '')
            
            # 调试：打印响应文本
            self.logger.debug(f"章节{section_index}生成响应: {response_text}")
            
            # 解析 HTML (传入 valid_urls 和 ref_url_mapping)
            parsed = self._parse_section_html(response_text, valid_urls, ref_url_mapping)
            
            if not parsed.get('content') and not parsed.get('html'):
                return {
                    'success': False,
                    'content': '',
                    'html': '',
                    'references': [],
                    'error_message': '章节内容解析失败'
                }
            
            return {
                'success': True,
                'content': parsed['content'],
                'html': parsed.get('html', ''),
                'references': parsed['references'],
                'error_message': ''
            }
            
        except Exception as e:
            return {
                'success': False,
                'content': '',
                'references': [],
                'error_message': str(e)
            }
    
    def _parse_section_html(self, xml_text: str, valid_urls: Set[str] = None, ref_url_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """解析章节 HTML（之前是XML，现已改为HTML格式）"""
        try:
            from bs4 import BeautifulSoup
            import re
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(xml_text, 'html.parser')
            
            # 查找article标签
            article = soup.find('article')
            if not article:
                self.logger.warning("⚠️ 未找到 <article> 标签")
                return {'content': '', 'references': [], 'html': ''}
            
            # 提取正文内容（blockquote部分）
            content_block = soup.find('blockquote', {'data-field': 'content'})
            if not content_block:
                self.logger.warning("⚠️ 未找到正文内容")
                return {'content': '', 'references': [], 'html': ''}
            
            # 提取纯文本内容（用于字数统计等）
            content_text = content_block.get_text(strip=True)
            
            # 提取参考文献
            references = []
            footer = soup.find('footer')
            if footer:
                references_list = footer.find('ol', class_='references')
                if references_list:
                    for item in references_list.find_all('li'):
                        ref_id = item.get('data-ref-id', '')
                        url_span = item.find('span', {'data-field': 'url'})
                        desc_span = item.find('span', {'data-field': 'description'})
                        
                        if ref_id and url_span:
                            url = url_span.get_text(strip=True)
                            description = desc_span.get_text(strip=True) if desc_span else f'参考文献{ref_id}'
                            
                            references.append({
                                'id': ref_id,
                                'url': url,
                                'description': description
                            })
            
            # 如果没有找到references但内容中有引用标记，尝试重建references
            if not references and content_block and ref_url_mapping:
                self.logger.info("⚠️ LLM未输出<footer>，从引用标记重建引用")
                # 查找所有 <sup data-cite="n"> 标签
                cite_sups = content_block.find_all('sup', attrs={'data-cite': True})
                cite_refs = set()
                for sup in cite_sups:
                    ref_id = sup.get('data-cite', '')
                    if ref_id:
                        cite_refs.add(ref_id)
                
                for ref_id in sorted(cite_refs, key=lambda x: int(x) if x.isdigit() else 0):
                    if ref_id in ref_url_mapping:
                        url = ref_url_mapping[ref_id]
                        references.append({
                            'id': ref_id,
                            'url': url,
                            'description': f'参考文献{ref_id}'
                        })
                        self.logger.debug(f"  重建引用 [{ref_id}]: {url[:80]}...")
                
                if references:
                    self.logger.info(f"✓ 成功重建 {len(references)} 个引用")
            
            return {
                'content': content_text,  # 纯文本内容
                'html': str(article),     # 完整HTML
                'references': references
            }
            
        except Exception as e:
            self.logger.error(f"❌ 解析章节 HTML 时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'content': '', 'html': '', 'references': []}

    def _assemble_article(self, task: str, sections_data: List[Dict]) -> Dict[str, Any]:
        """组装完整文章（HTML格式）"""
        try:
            from bs4 import BeautifulSoup
            import re
            
            # 统一处理参考文献编号
            article_html_parts = []
            article_text_parts = []
            all_references = {}
            ref_counter = 1
            
            # 开始构建HTML文章
            article_html_parts.append('<article data-report="true">')
            article_html_parts.append(f'  <header>')
            article_html_parts.append(f'    <h1 data-field="title">研究报告：{task}</h1>')
            article_html_parts.append(f'  </header>')
            article_html_parts.append(f'  <main>')
            
            # 纯文本版本的标题
            article_text_parts.append(f"# 研究报告：{task}\n\n")
            
            # 处理每个章节
            for section in sections_data:
                section_title = section['title']
                section_html = section.get('html', '')  # HTML格式的章节内容
                section_refs = section['references']
                
                # 解析章节HTML
                soup = BeautifulSoup(section_html, 'html.parser')
                
                # 构建引用映射（将章节内的局部编号映射到全局编号）
                ref_mapping = {}
                for ref in section_refs:
                    old_id = ref['id']
                    ref_mapping[old_id] = str(ref_counter)
                    all_references[str(ref_counter)] = {
                        'url': ref['url'],
                        'description': ref['description']
                    }
                    ref_counter += 1
                
                # 替换引用编号：<sup data-cite="old_id"> → <sup data-cite="new_id">
                # 找到所有引用标记并替换
                for sup in soup.find_all('sup', attrs={'data-cite': True}):
                    old_cite = sup.get('data-cite', '')
                    if old_cite in ref_mapping:
                        sup['data-cite'] = ref_mapping[old_cite]
                
                # 提取正文内容（blockquote部分）
                content_block = soup.find('blockquote', {'data-field': 'content'})
                if content_block:
                    # 添加章节到HTML
                    article_html_parts.append(f'    <section>')
                    article_html_parts.append(f'      <h2>{section_title}</h2>')
                    # 提取content_block的内部HTML
                    content_html = ''.join(str(child) for child in content_block.children)
                    article_html_parts.append(f'      {content_html}')
                    article_html_parts.append(f'    </section>')
                    
                    # 纯文本版本：转换引用标记为[n]
                    article_text_parts.append(f"## {section_title}\n\n")
                    for element in content_block.find_all(['h3', 'p']):
                        if element.name == 'h3':
                            article_text_parts.append(f"### {element.get_text(strip=True)}\n\n")
                        elif element.name == 'p':
                            # 转换引用标记
                            text = self._convert_html_citations_to_text(element)
                            article_text_parts.append(f"{text}\n\n")
            
            # 添加参考文献部分（HTML）
            article_html_parts.append(f'  </main>')
            article_html_parts.append(f'  <footer>')
            article_html_parts.append(f'    <h2>参考文献</h2>')
            article_html_parts.append(f'    <ol class="references">')
            
            # HTML版本：保持原样，每个引用ID一条
            for ref_id in sorted(all_references.keys(), key=int):
                ref = all_references[ref_id]
                article_html_parts.append(f'      <li data-ref-id="{ref_id}">')
                article_html_parts.append(f'        <cite>')
                article_html_parts.append(f'          <span data-field="url">{ref["url"]}</span>')
                article_html_parts.append(f'          <span data-field="description">{ref["description"]}</span>')
                article_html_parts.append(f'        </cite>')
                article_html_parts.append(f'      </li>')
            
            article_html_parts.append(f'    </ol>')
            article_html_parts.append(f'  </footer>')
            article_html_parts.append(f'</article>')
            
            # 纯文本版本：按URL去重合并参考文献
            if all_references:
                # 按URL分组：{url: [ref_ids]}
                url_to_refs = {}
                for ref_id in sorted(all_references.keys(), key=int):
                    ref = all_references[ref_id]
                    url = ref['url']
                    if url not in url_to_refs:
                        url_to_refs[url] = []
                    url_to_refs[url].append({
                        'id': ref_id,
                        'description': ref['description']
                    })
                
                # 构建去重后的纯文本参考文献列表
                refs_text_parts = []
                text_ref_counter = 1
                
                for url, refs_list in url_to_refs.items():
                    # 合并描述（如果有多个相同URL的引用）
                    if len(refs_list) == 1:
                        # 只有一个引用，直接使用
                        desc = refs_list[0]['description']
                    else:
                        # 多个引用，合并描述（去重）
                        descriptions = list(dict.fromkeys([r['description'] for r in refs_list]))
                        if len(descriptions) == 1:
                            desc = descriptions[0]
                        else:
                            # 多个不同描述，合并为一个
                            desc = " / ".join(descriptions)
                    
                    refs_text_parts.append(f"[{text_ref_counter}] {desc}. {url}")
                    text_ref_counter += 1
                
                # 添加到纯文本版本
                article_text_parts.append("\n## 参考文献\n\n")
                article_text_parts.append('\n'.join(refs_text_parts))
            
            return {
                'success': True,
                'article_html': '\n'.join(article_html_parts),  # 保持字段名兼容性
                'article_text': ''.join(article_text_parts),
                'references_count': len(all_references),
                'error_message': ''
            }
            
        except Exception as e:
            self.logger.error(f"❌ 组装文章失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'article_html': '',
                'article_text': '',
                'references_count': 0,
                'error_message': str(e)
            }
    
    def _convert_html_citations_to_text(self, element) -> str:
        """
        将HTML元素中的引用标记转换为纯文本格式
        与test_extract_html.py中的_convert_citations_to_text功能相同
        """
        from bs4 import BeautifulSoup
        
        # 创建元素的副本以避免修改原始元素
        element_copy = BeautifulSoup(str(element), 'html.parser')
        
        # 处理 <cite ref="n"> 标签（旧格式兼容）
        for cite in element_copy.find_all('cite'):
            ref = cite.get('ref', '')
            if ref:
                cite.replace_with(f'[{ref}]')
        
        # 处理 <sup data-cite="n"> 标签（新格式）
        for sup in element_copy.find_all('sup'):
            cite_ref = sup.get('data-cite', '')
            if cite_ref:
                sup.replace_with(f'[{cite_ref}]')
        
        return element_copy.get_text(strip=True)
    
    def _format_knowledge_for_prompt(self, knowledge_list: List[Dict]) -> str:
        """
        格式化知识列表为引用友好的提示词文本
        
        将知识分为：
        - A类：事实/研究共识型知识（无URL，不可引用）
        - B类：明确提供了真实URL的知识（可引用文献）
        """
        if not knowledge_list:
            return "暂无相关知识"
        
        # 分类知识
        citable_knowledge = []  # B类：可引用（有URL）
        non_citable_knowledge = []  # A类：不可引用（无URL）
        
        for idx, item in enumerate(knowledge_list, 1):
            content = item.get('content', '')
            distance = item.get('distance', 0)
            metadata = item.get('metadata', {})
            source = metadata.get('source_title', '未知来源')
            url = metadata.get('url') or metadata.get('source_url', '')  # 兼容两种字段名
            
            if url and url.strip() and not url.startswith('unknown'):
                # B类：可引用文献
                citable_knowledge.append({
                    'content': content,
                    'distance': distance,
                    'url': url,
                    'source': source
                })
            else:
                # A类：不可引用的研究共识
                non_citable_knowledge.append({
                    'content': content,
                    'distance': distance,
                    'source': source
                })
        
        # 构建引用友好的格式化文本
        formatted_parts = []
        
        # B类：可引用文献（白名单）
        if citable_knowledge:

            for idx, item in enumerate(citable_knowledge, 1):
                formatted_parts.append(f"data-ref-id=\"{idx}\"")
                formatted_parts.append(f"URL: {item['url']}")
                formatted_parts.append(f"Content: {item['content']}")

                formatted_parts.append("")
        # 对于non_citable_knowledge 而言，不使用
        non_citable_knowledge = None
        # A类：不可引用的研究共识
        if non_citable_knowledge:
            formatted_parts.append("### B. 不可引用的研究共识与事实（仅用于分析，不得引用）")
            formatted_parts.append("")
            formatted_parts.append("以下内容为内部知识库整理结果，没有公开可核验 URL，")
            formatted_parts.append("仅可作为分析与判断依据，不得生成引用编号。")
            formatted_parts.append("")
            
            for idx, item in enumerate(non_citable_knowledge, 1):
                formatted_parts.append(f"共识 {idx}:")
                formatted_parts.append(f"  内容: {item['content']}")

                formatted_parts.append("")

        return '\n'.join(formatted_parts)
    
    def _remove_xml_tags(self, text: str) -> str:
        """移除文本中的 XML 标签"""
        import re
        # 移除所有 XML 标签
        clean_text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空行
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        return clean_text.strip()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, mbti={self.mbti_type})"
