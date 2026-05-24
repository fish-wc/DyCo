"""
配置数据结构定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """大模型配置"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="模型名称")
    model_url: str = Field(..., description="模型API地址")
    api_key: str = Field(..., description="API密钥")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    timeout: int = Field(default=60, description="超时时间(秒)")
    
    # 如下参数是针对 smolagent配置的
    max_steps:int = Field(default=10, description="最大步骤数")
    verbosity_level:int = Field(default=0, description="冗长度级别")

    # 以下是针对embedding模型的配置
    dimension: Optional[int] = Field(default=None, description="嵌入维度")

class KnowledgeConfig(BaseModel):
    """知识库配置"""
    private_kb_path: str = Field(..., description="私有知识库路径")
    shared_kb_path: str = Field(..., description="共享知识库路径")
    vector_store_type: str = Field(default="faiss", description="向量存储类型")
    embedding_model: Optional[str] = Field(None, description="嵌入模型")


class PersonalityConfig(BaseModel):
    """性格配置"""
    mbti_type: Optional[str] = Field(None, description="MBTI类型")
    speaking_threshold: float = Field(default=6.0, ge=0.0, le=10.0, description="发言阈值(0-10分)")


class AgentConfig(BaseModel):
    """智能体配置"""
    agent_id: str = Field(..., description="智能体ID")
    agent_name: str = Field(..., description="智能体名称")
    mbti_type: str = Field(..., description="MBTI类型")
    
    # 模型配置
    model: ModelConfig = Field(..., description="大模型配置")
    
    # 性格配置
    personality: PersonalityConfig = Field(..., description="性格配置")
    
    
    # 知识库配置
    knowledge: KnowledgeConfig = Field(..., description="知识库配置")
    
    # 其他配置
    max_memory_items: int = Field(default=100, description="最大记忆条目数")
    workspace_path: str = Field(..., description="工作空间路径")
    memory_window_size: int = Field(default=6, description="记忆窗口大小")

class SystemConfig(BaseModel):
    """系统配置"""
    workspace_root: str = Field(..., description="工作空间根目录")
    message_storage_type: str = Field(default="json", description="消息存储类型")
    log_level: str = Field(default="INFO", description="日志级别")
    max_concurrent_agents: int = Field(default=4, description="最大并发智能体数")
    max_evaluation_agents: int = Field(default=4, description="最大评测智能体数")
    max_rounds: int = Field(default=10, description="最大交互轮数")
    max_final_discussion_rounds:int = Field(default=15,description="最后一轮讨论")
    collate_memory_window: int = Field(default=10, description="总结记忆窗口大小")
    attitude_memory_window: int = Field(default=2, description="态度记忆窗口大小")
    web_search_count: int = Field(default=2, description="网络搜索结果数量")
    num_listeners_sampled_per_round: int = Field(default=1, description="每轮抽样评估共识的听众数量")
    agents: List[Dict[str, str]] = Field(default_factory=list, description="智能体列表")

    # 模型配置
    model: ModelConfig = Field(..., description="大模型配置")
    embedding: ModelConfig = Field(..., description="嵌入模型")
    collate_model: Optional[ModelConfig] = Field(None, description="Collate专用大模型配置（用于整理最终文章）")