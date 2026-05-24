"""
消息数据结构定义
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class MessageType(str, Enum):
    """消息类型"""
    EVALUATION = "evaluation"  # 评判/验证阶段，用于初始方案评测
    DISCUSSION = "discussion"  # 讨论发言
    TEAM_FORMATION = "team_formation"  # 组队意向
    SOLUTION = "solution"  # 解决方案
    FINAL_ANSWER = "final_answer"  # 最终答案
    INTERNAL_MEMO = "internal_memo"  # 内部备忘


class RoundStage(str, Enum):
    """讨论轮次"""
    INITIAL_SOLUTION = "initial_solution"  # 初始解决阶段
    ROUND_1_DISCUSSION = "round_1_discussion"  # 第一轮:全体讨论
    ROUND_2_TEAM_FORMATION = "round_2_team_formation"  # 第二轮:组队
    ROUND_3_TEAM_DISCUSSION = "round_3_team_discussion"  # 第三轮:小组讨论
    ROUND_4_FINAL_DISCUSSION = "round_4_final_discussion"  # 第四轮:代表讨论


class Message(BaseModel):
    """消息数据结构"""
    message_id: str = Field(..., description="消息唯一标识")
    sender_id: str = Field(..., description="发言主体ID(智能体名称)")
    team_id: str = Field(..., description="团队ID(智能体名称按字典序排列)")
    round_stage: RoundStage = Field(..., description="讨论轮次")
    sequence: int = Field(..., description="发言顺序")
    message_type: MessageType = Field(default=MessageType.DISCUSSION, description="消息类型")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MessageBatch(BaseModel):
    """消息批次,用于批量存储和读取消息"""
    task_id: str = Field(..., description="任务ID")
    messages: List[Message] = Field(default_factory=list, description="消息列表")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TeamFormation(BaseModel):
    """组队意向数据结构"""
    agent_id: str = Field(..., description="智能体ID")
    preferred_teammates: List[str] = Field(default_factory=list, description="期望的队友列表")
    reason: str = Field(..., description="组队理由")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SpeakingIntention(BaseModel):
    """发言意愿数据结构"""
    agent_id: str = Field(..., description="智能体ID")
    intention_score: float = Field(default=0.0, ge=0.0, le=10.0, description="发言意愿分数(0-10)")
    threshold: float = Field(default=6.0, ge=0.0, le=10.0, description="发言阈值(0-10)")
    context: Optional[str] = Field(None, description="触发发言意愿的上下文")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
    @property
    def should_speak(self) -> bool:
        """是否应该发言"""
        return self.intention_score >= self.threshold
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
