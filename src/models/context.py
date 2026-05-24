"""
上下文数据传输类型 - 通用数据管道
参考 MetaGPT 和 LangChain 的设计理念,实现项目内部的通用数据流转结构

设计理念:
1. 数据不可变性 (Immutability): 每次更新返回新对象,避免意外修改
2. 类型安全 (Type Safety): 使用 Pydantic 强类型验证
3. 可扩展性 (Extensibility): 通过 metadata 和 extras 字段支持自定义扩展
4. 可追溯性 (Traceability): 记录数据流转的完整生命周期

核心数据流:
task -> task_context -> llm_request -> llm_response -> tool_execution -> discussion_context -> final_result
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# ==================== 基础枚举类型 ====================

class ContextType(str, Enum):
    """上下文类型枚举"""
    TASK = "task"                           # 任务上下文
    LLM_REQUEST = "llm_request"             # LLM请求上下文
    LLM_RESPONSE = "llm_response"           # LLM响应上下文
    TOOL_EXECUTION = "tool_execution"       # 工具执行上下文
    AGENT_ACTION = "agent_action"           # Agent行动上下文
    DISCUSSION = "discussion"               # 讨论上下文
    WORKFLOW = "workflow"                   # 工作流上下文


class ExecutionStatus(str, Enum):
    """执行状态枚举"""
    PENDING = "pending"           # 等待执行
    RUNNING = "running"           # 执行中
    SUCCESS = "success"           # 执行成功
    FAILED = "failed"             # 执行失败
    CANCELLED = "cancelled"       # 已取消
    TIMEOUT = "timeout"           # 超时


# ==================== 基础上下文类 ====================

class BaseContext(BaseModel):
    """
    基础上下文类 - 所有上下文的父类
    
    提供通用的上下文管理功能:
    - 唯一标识
    - 时间戳记录
    - 元数据存储
    - 父子关系追踪
    """
    context_id: str = Field(..., description="上下文唯一标识")
    context_type: ContextType = Field(..., description="上下文类型")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    # 父子关系追踪
    parent_context_id: Optional[str] = Field(None, description="父上下文ID,用于追踪数据流转")
    
    # 元数据和扩展字段
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据,存储额外信息")
    extras: Dict[str, Any] = Field(default_factory=dict, description="扩展字段,用于自定义数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_metadata(self, key: str, value: Any) -> "BaseContext":
        """
        更新元数据 (返回新对象,保持不可变性)
        
        Args:
            key: 元数据键
            value: 元数据值
            
        Returns:
            更新后的新上下文对象
        """
        new_context = self.model_copy(deep=True)
        new_context.metadata[key] = value
        new_context.updated_at = datetime.now()
        return new_context


# ==================== 任务上下文 ====================

class TaskContext(BaseContext):
    """
    任务上下文 - 工作流的起点
    
    用途:
    - 封装用户输入的原始任务
    - 传递任务相关配置
    - 记录任务执行状态
    
    使用场景:
    - workflow.run(task) -> TaskContext
    - agent.analyze_task(task_context)
    """
    context_type: ContextType = Field(default=ContextType.TASK, description="固定为TASK类型")
    
    # 任务基本信息
    task_id: str = Field(..., description="任务ID")
    task_description: str = Field(..., description="任务描述")
    task_requirements: Optional[str] = Field(None, description="任务要求")
    
    # 任务配置
    priority: int = Field(default=5, ge=1, le=10, description="优先级(1-10)")
    max_rounds: Optional[int] = Field(None, description="最大执行轮数")
    timeout: Optional[int] = Field(None, description="超时时间(秒)")
    
    # 任务状态
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="执行状态")
    
    # 任务结果
    result: Optional[str] = Field(None, description="任务执行结果")
    error_message: Optional[str] = Field(None, description="错误信息")


# ==================== LLM 相关上下文 ====================

class LLMRequestContext(BaseContext):
    """
    LLM请求上下文
    
    用途:
    - 封装所有LLM调用的输入参数
    - 统一管理 system_prompt, user_prompt, tools 等
    - 便于日志记录和调试
    
    使用场景:
    - agent.analyze_task() -> LLMRequestContext -> call_llm()
    - workflow 中的各种 LLM 调用
    
    优势:
    - 替代原来的多个散乱参数 (system_prompt, user_prompt, temperature...)
    - 支持工具调用的完整配置
    - 便于重试和缓存
    """
    context_type: ContextType = Field(default=ContextType.LLM_REQUEST, description="固定为LLM_REQUEST类型")
    
    # 基本参数
    model_name: str = Field(..., description="模型名称")
    system_prompt: str = Field(default="", description="系统提示词")
    user_prompt: str = Field(..., description="用户提示词")
    
    # 生成参数
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p采样")
    
    # 工具调用配置
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="可用工具列表")
    tool_choice: Optional[str] = Field(None, description="工具选择策略: auto/required/none")
    
    # 历史对话 (用于多轮对话)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="历史对话记录")
    
    # 请求标识
    request_id: Optional[str] = Field(None, description="请求ID,用于追踪")
    caller_id: Optional[str] = Field(None, description="调用者ID (agent_id 或 workflow_id)")
    
    # 日志配置
    logger_name: str = Field(default="", description="日志记录器名称")
    
    def add_message(self, role: str, content: str) -> "LLMRequestContext":
        """添加对话消息"""
        new_context = self.model_copy(deep=True)
        new_context.conversation_history.append({"role": role, "content": content})
        new_context.updated_at = datetime.now()
        return new_context


class LLMResponseContext(BaseContext):
    """
    LLM响应上下文
    
    用途:
    - 封装 LLM 返回的所有信息
    - 包含生成文本、token统计、工具调用等
    - 记录执行时长和状态
    
    使用场景:
    - call_llm() 返回 LLMResponseContext 而不是纯文本
    - 便于后续处理和分析
    """
    context_type: ContextType = Field(default=ContextType.LLM_RESPONSE, description="固定为LLM_RESPONSE类型")
    
    # 关联的请求
    request_context_id: str = Field(..., description="关联的请求上下文ID")
    
    # 响应内容
    response_text: str = Field(..., description="生成的文本内容")
    finish_reason: Optional[str] = Field(None, description="结束原因: stop/length/tool_calls")
    
    # Token统计
    prompt_tokens: int = Field(default=0, description="输入token数")
    completion_tokens: int = Field(default=0, description="输出token数")
    total_tokens: int = Field(default=0, description="总token数")
    
    # 工具调用结果
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="工具调用记录")
    
    # 执行信息
    status: ExecutionStatus = Field(default=ExecutionStatus.SUCCESS, description="执行状态")
    elapsed_time: float = Field(default=0.0, description="执行耗时(秒)")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 原始响应 (可选,用于调试)
    raw_response: Optional[Dict[str, Any]] = Field(None, description="原始API响应")


# ==================== 工具执行上下文 ====================

class ToolExecutionContext(BaseContext):
    """
    工具执行上下文
    
    用途:
    - 封装工具调用的输入输出
    - 记录执行状态和耗时
    - 支持嵌套工具调用
    
    使用场景:
    - web_search_tool() 返回 ToolExecutionContext
    - knowledge_tool() 返回 ToolExecutionContext
    - LLMToolClient.call_with_tools() 返回包含多个 ToolExecutionContext 的列表
    """
    context_type: ContextType = Field(default=ContextType.TOOL_EXECUTION, description="固定为TOOL_EXECUTION类型")
    
    # 工具信息
    tool_name: str = Field(..., description="工具名称")
    tool_description: Optional[str] = Field(None, description="工具描述")
    
    # 输入输出
    arguments: Dict[str, Any] = Field(..., description="工具调用参数")
    result: Any = Field(..., description="工具执行结果")
    result_preview: Optional[str] = Field(None, description="结果预览(截断显示)")
    
    # 执行信息
    status: ExecutionStatus = Field(default=ExecutionStatus.SUCCESS, description="执行状态")
    elapsed_time: float = Field(default=0.0, description="执行耗时(秒)")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 调用链追踪
    caller_id: Optional[str] = Field(None, description="调用者ID")
    call_depth: int = Field(default=0, description="调用深度(嵌套层级)")
    
    # 资源消耗统计 (如JINA token)
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="资源使用统计")


# ==================== Agent 行动上下文 ====================

class AgentActionContext(BaseContext):
    """
    Agent行动上下文
    
    用途:
    - 封装 Agent 执行的单个行动 (分析任务、评估方案、组队意愿等)
    - 关联输入、输出、使用的资源
    - 便于追踪 Agent 行为链
    
    使用场景:
    - agent.analyze_task(task_context) -> AgentActionContext
    - agent.evaluate_solution(task_context, solution_context) -> AgentActionContext
    - agent.decide_team_preference() -> AgentActionContext
    """
    context_type: ContextType = Field(default=ContextType.AGENT_ACTION, description="固定为AGENT_ACTION类型")
    
    # Agent信息
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent名称")
    mbti_type: Optional[str] = Field(None, description="MBTI类型")
    
    # 行动信息
    action_type: str = Field(..., description="行动类型: analyze_task/evaluate_solution/decide_team_preference/speak/generate等")
    action_description: Optional[str] = Field(None, description="行动描述")
    
    # 输入输出
    input_context: Optional[Dict[str, Any]] = Field(None, description="输入上下文")
    output_result: Any = Field(..., description="输出结果")
    
    # 资源使用
    llm_requests: List[str] = Field(default_factory=list, description="关联的LLM请求ID列表")
    tool_executions: List[str] = Field(default_factory=list, description="关联的工具执行ID列表")
    
    # 执行信息
    status: ExecutionStatus = Field(default=ExecutionStatus.SUCCESS, description="执行状态")
    elapsed_time: float = Field(default=0.0, description="执行耗时(秒)")
    error_message: Optional[str] = Field(None, description="错误信息")


# ==================== 讨论上下文 ====================

class DiscussionContext(BaseContext):
    """
    讨论上下文
    
    用途:
    - 封装多个 Agent 的讨论过程
    - 管理发言顺序、团队信息、讨论轮次
    - 聚合所有 Agent 的行动上下文
    
    使用场景:
    - workflow.round_1_all_discussion(task_context) -> DiscussionContext
    - workflow.round_3_team_discussion(task_context, teams) -> DiscussionContext
    - workflow.round_4_final_discussion(task_context, representatives) -> DiscussionContext
    """
    context_type: ContextType = Field(default=ContextType.DISCUSSION, description="固定为DISCUSSION类型")
    
    # 讨论基本信息
    discussion_id: str = Field(..., description="讨论ID")
    task_context_id: str = Field(..., description="关联的任务上下文ID")
    round_stage: str = Field(..., description="讨论轮次")
    
    # 参与者信息
    participants: List[str] = Field(..., description="参与者ID列表")
    team_id: Optional[str] = Field(None, description="团队ID (小组讨论时使用)")
    
    # 讨论记录
    agent_actions: List[str] = Field(default_factory=list, description="Agent行动上下文ID列表")
    speaking_sequence: List[str] = Field(default_factory=list, description="发言顺序(agent_id列表)")
    
    # 讨论结果
    consensus_reached: bool = Field(default=False, description="是否达成共识")
    final_output: Optional[str] = Field(None, description="讨论最终输出")
    
    # 统计信息
    total_rounds: int = Field(default=0, description="总发言轮数")
    total_llm_calls: int = Field(default=0, description="总LLM调用次数")
    total_tokens: int = Field(default=0, description="总token消耗")
    
    # 执行信息
    status: ExecutionStatus = Field(default=ExecutionStatus.RUNNING, description="执行状态")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    finished_at: Optional[datetime] = Field(None, description="结束时间")


# ==================== 工作流上下文 ====================

class WorkflowContext(BaseContext):
    """
    工作流上下文 - 最顶层的数据结构
    
    用途:
    - 管理整个工作流的执行过程
    - 聚合所有子上下文 (任务、讨论、Agent行动等)
    - 提供全局视图和追踪能力
    
    使用场景:
    - workflow.run(task) -> WorkflowContext
    - 包含所有阶段的上下文ID引用
    """
    context_type: ContextType = Field(default=ContextType.WORKFLOW, description="固定为WORKFLOW类型")
    
    # 工作流基本信息
    workflow_id: str = Field(..., description="工作流ID")
    workflow_type: str = Field(..., description="工作流类型: four_round_discussion/simple_task/multi_agent_collaboration等")
    
    # 关联的上下文
    task_context_id: str = Field(..., description="关联的任务上下文ID")
    discussion_contexts: List[str] = Field(default_factory=list, description="讨论上下文ID列表")
    agent_actions: List[str] = Field(default_factory=list, description="所有Agent行动ID列表")
    
    # 工作流阶段
    current_stage: str = Field(default="initialized", description="当前阶段")
    completed_stages: List[str] = Field(default_factory=list, description="已完成阶段列表")
    
    # 工作流结果
    final_result: Optional[str] = Field(None, description="最终结果")
    
    # 统计信息
    total_agents: int = Field(default=0, description="参与Agent总数")
    total_discussions: int = Field(default=0, description="讨论次数")
    total_llm_calls: int = Field(default=0, description="LLM调用总次数")
    total_tokens: int = Field(default=0, description="token消耗总量")
    total_elapsed_time: float = Field(default=0.0, description="总耗时(秒)")
    
    # 执行信息
    status: ExecutionStatus = Field(default=ExecutionStatus.RUNNING, description="执行状态")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    finished_at: Optional[datetime] = Field(None, description="结束时间")


# ==================== 上下文管理器 ====================

class ContextManager:
    """
    上下文管理器 - 管理所有上下文对象的生命周期
    
    功能:
    - 统一存储和检索上下文对象
    - 维护上下文之间的关联关系
    - 提供查询和统计功能
    
    使用示例:
    ```python
    # 初始化管理器
    manager = ContextManager()
    
    # 创建并注册任务上下文
    task_ctx = TaskContext(
        context_id="task_001",
        task_id="task_001",
        task_description="分析量子计算的发展趋势"
    )
    manager.register(task_ctx)
    
    # 创建LLM请求上下文(关联到任务)
    llm_req = LLMRequestContext(
        context_id="llm_req_001",
        parent_context_id="task_001",
        model_name="gpt-4",
        user_prompt="请分析量子计算..."
    )
    manager.register(llm_req)
    
    # 检索
    task = manager.get("task_001")
    children = manager.get_children("task_001")  # 获取所有子上下文
    ```
    """
    
    def __init__(self):
        self._contexts: Dict[str, BaseContext] = {}
        self._type_index: Dict[ContextType, List[str]] = {}
        self._parent_index: Dict[str, List[str]] = {}
    
    def register(self, context: BaseContext) -> None:
        """
        注册上下文对象
        
        Args:
            context: 上下文对象
        """
        context_id = context.context_id
        self._contexts[context_id] = context
        
        # 更新类型索引
        if context.context_type not in self._type_index:
            self._type_index[context.context_type] = []
        self._type_index[context.context_type].append(context_id)
        
        # 更新父子索引
        if context.parent_context_id:
            if context.parent_context_id not in self._parent_index:
                self._parent_index[context.parent_context_id] = []
            self._parent_index[context.parent_context_id].append(context_id)
    
    def get(self, context_id: str) -> Optional[BaseContext]:
        """获取上下文对象"""
        return self._contexts.get(context_id)
    
    def get_by_type(self, context_type: ContextType) -> List[BaseContext]:
        """根据类型获取所有上下文"""
        context_ids = self._type_index.get(context_type, [])
        return [self._contexts[cid] for cid in context_ids]
    
    def get_children(self, parent_context_id: str) -> List[BaseContext]:
        """获取所有子上下文"""
        child_ids = self._parent_index.get(parent_context_id, [])
        return [self._contexts[cid] for cid in child_ids]
    
    def get_root_contexts(self) -> List[BaseContext]:
        """获取所有根上下文(没有父节点的上下文)"""
        return [ctx for ctx in self._contexts.values() if ctx.parent_context_id is None]
    
    def clear(self) -> None:
        """清空所有上下文"""
        self._contexts.clear()
        self._type_index.clear()
        self._parent_index.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_contexts": len(self._contexts),
            "by_type": {
                ctx_type.value: len(ctx_ids) 
                for ctx_type, ctx_ids in self._type_index.items()
            },
            "root_contexts": len(self.get_root_contexts())
        }


# ==================== 工具函数 ====================

def create_context_id(prefix: str = "ctx") -> str:
    """
    生成上下文ID
    
    Args:
        prefix: ID前缀
        
    Returns:
        唯一的上下文ID
    """
    from uuid import uuid4
    return f"{prefix}_{uuid4().hex[:12]}"


def link_contexts(parent: BaseContext, child: BaseContext) -> BaseContext:
    """
    建立父子上下文关系
    
    Args:
        parent: 父上下文
        child: 子上下文
        
    Returns:
        更新后的子上下文
    """
    new_child = child.model_copy(deep=True)
    new_child.parent_context_id = parent.context_id
    new_child.updated_at = datetime.now()
    return new_child
