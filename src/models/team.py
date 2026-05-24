"""
团队数据结构定义
"""
from typing import List, Set, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TeamMember(BaseModel):
    """团队成员"""
    agent_id: str = Field(..., description="智能体ID")
    role: str = Field(default="member", description="角色: member/representative")
    join_time: datetime = Field(default_factory=datetime.now, description="加入时间")


class Team(BaseModel):
    """团队数据结构"""
    team_id: str = Field(..., description="团队ID(成员名按字典序排列)")
    members: List[TeamMember] = Field(default_factory=list, description="团队成员列表")
    round_stage: str = Field(..., description="所属轮次")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    metadata: Dict = Field(default_factory=dict, description="元数据")
    
    @staticmethod
    def generate_team_id(agent_ids: List[str]) -> str:
        """生成团队ID: 将智能体ID按字典序排序后连接"""
        sorted_ids = sorted(agent_ids)
        return "_".join(sorted_ids)
    
    @property
    def member_ids(self) -> List[str]:
        """获取所有成员ID"""
        return [member.agent_id for member in self.members]
    
    def is_member(self, agent_id: str) -> bool:
        """判断某个智能体是否是团队成员"""
        return agent_id in self.member_ids
    
    def get_representative(self) -> Optional[str]:
        """获取团队代表"""
        for member in self.members:
            if member.role == "representative":
                return member.agent_id
        return None
    
    def set_representative(self, agent_id: str):
        """设置团队代表"""
        if not self.is_member(agent_id):
            raise ValueError(f"Agent {agent_id} is not a member of this team")
        
        # 清除所有代表角色
        for member in self.members:
            if member.role == "representative":
                member.role = "member"
        
        # 设置新代表
        for member in self.members:
            if member.agent_id == agent_id:
                member.role = "representative"
                break
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TeamRegistry(BaseModel):
    """团队注册表,管理所有团队"""
    task_id: str = Field(..., description="任务ID")
    teams: Dict[str, Team] = Field(default_factory=dict, description="团队字典: team_id -> Team")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    def add_team(self, team: Team):
        """添加团队"""
        self.teams[team.team_id] = team
        self.updated_at = datetime.now()
    
    def get_team(self, team_id: str) -> Optional[Team]:
        """获取团队"""
        return self.teams.get(team_id)
    
    def get_agent_teams(self, agent_id: str) -> List[Team]:
        """获取智能体所属的所有团队"""
        return [team for team in self.teams.values() if team.is_member(agent_id)]
    
    def get_teams_by_round(self, round_stage: str) -> List[Team]:
        """获取某个轮次的所有团队"""
        return [team for team in self.teams.values() if team.round_stage == round_stage]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
