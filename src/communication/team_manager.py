"""
团队管理器
管理智能体的组队和团队信息
"""
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from ..models.team import Team, TeamMember, TeamRegistry
from ..models.message import RoundStage


class TeamManager:
    """团队管理器"""
    
    def __init__(self, storage_path: str):
        """
        初始化团队管理器
        
        Args:
            storage_path: 团队信息存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._registries: Dict[str, TeamRegistry] = {}
    
    def _get_registry_file_path(self, task_id: str) -> Path:
        """获取团队注册表文件路径 - 每个任务一个文件夹"""
        task_folder = self.storage_path / task_id  # task_id 已包含 "task_" 前缀
        task_folder.mkdir(parents=True, exist_ok=True)
        return task_folder / "teams.json"
    
    def _load_registry(self, task_id: str) -> TeamRegistry:
        """加载团队注册表"""
        if task_id in self._registries:
            return self._registries[task_id]
        
        registry_file = self._get_registry_file_path(task_id)
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                registry = TeamRegistry(**data)
        else:
            registry = TeamRegistry(task_id=task_id)
        
        self._registries[task_id] = registry
        return registry
    
    def _save_registry(self, task_id: str):
        """保存团队注册表"""
        if task_id not in self._registries:
            return
        
        registry = self._registries[task_id]
        registry_file = self._get_registry_file_path(task_id)
        
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    
    def create_team(self, task_id: str, agent_ids: List[str], round_stage: RoundStage) -> Team:
        """
        创建团队
        
        Args:
            task_id: 任务ID
            agent_ids: 智能体ID列表
            round_stage: 轮次阶段
            
        Returns:
            创建的团队对象
        """
        team_id = Team.generate_team_id(agent_ids)
        members = [TeamMember(agent_id=aid) for aid in agent_ids]
        
        team = Team(
            team_id=team_id,
            members=members,
            round_stage=round_stage.value if isinstance(round_stage, RoundStage) else round_stage
        )
        
        registry = self._load_registry(task_id)
        registry.add_team(team)
        self._save_registry(task_id)
        
        return team
    
    def get_team(self, task_id: str, team_id: str) -> Optional[Team]:
        """
        获取团队信息
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            
        Returns:
            团队对象
        """
        registry = self._load_registry(task_id)
        return registry.get_team(team_id)
    
    def get_agent_teams(self, task_id: str, agent_id: str) -> List[Team]:
        """
        获取智能体所属的所有团队
        
        Args:
            task_id: 任务ID
            agent_id: 智能体ID
            
        Returns:
            团队列表
        """
        registry = self._load_registry(task_id)
        return registry.get_agent_teams(agent_id)
    
    def get_teams_by_round(self, task_id: str, round_stage: RoundStage) -> List[Team]:
        """
        获取某个轮次的所有团队
        
        Args:
            task_id: 任务ID
            round_stage: 轮次阶段
            
        Returns:
            团队列表
        """
        registry = self._load_registry(task_id)
        return registry.get_teams_by_round(round_stage.value if isinstance(round_stage, RoundStage) else round_stage)
    
    def set_team_representative(self, task_id: str, team_id: str, agent_id: str):
        """
        设置团队代表
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            agent_id: 智能体ID
        """
        registry = self._load_registry(task_id)
        team = registry.get_team(team_id)
        
        if team:
            team.set_representative(agent_id)
            self._save_registry(task_id)
    
    def get_team_representative(self, task_id: str, team_id: str) -> Optional[str]:
        """
        获取团队代表
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            
        Returns:
            代表的agent_id
        """
        team = self.get_team(task_id, team_id)
        return team.get_representative() if team else None
    
    def is_team_member(self, task_id: str, team_id: str, agent_id: str) -> bool:
        """
        判断智能体是否是团队成员
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            agent_id: 智能体ID
            
        Returns:
            是否是团队成员
        """
        team = self.get_team(task_id, team_id)
        return team.is_member(agent_id) if team else False
    
    def get_all_teams(self, task_id: str) -> List[Team]:
        """
        获取任务的所有团队
        
        Args:
            task_id: 任务ID
            
        Returns:
            团队列表
        """
        registry = self._load_registry(task_id)
        return list(registry.teams.values())
    
    def create_initial_team(self, task_id: str, all_agent_ids: List[str]) -> Team:
        """
        创建初始全员团队(第一轮讨论使用)
        
        Args:
            task_id: 任务ID
            all_agent_ids: 所有智能体ID列表
            
        Returns:
            创建的团队对象
        """
        return self.create_team(task_id, all_agent_ids, RoundStage.ROUND_1_DISCUSSION)
