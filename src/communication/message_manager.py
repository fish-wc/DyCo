"""
消息管理器
负责消息的存储、读取和检索
"""
import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from ..models.message import Message, MessageBatch, RoundStage


class MessageManager:
    """消息管理器"""
    
    def __init__(self, storage_path: str, storage_type: str = "json"):
        """
        初始化消息管理器
        
        Args:
            storage_path: 消息存储路径
            storage_type: 存储类型(json/database),目前只支持json
        """
        self.storage_path = Path(storage_path)
        self.storage_type = storage_type
        self.storage_path.mkdir(parents=True, exist_ok=True)
         
    def _get_message_file_path(self, task_id: str) -> Path:
        """获取消息文件路径 - 每个任务一个文件夹"""
        task_folder = self.storage_path / task_id  # task_id 已包含 "task_" 前缀
        task_folder.mkdir(parents=True, exist_ok=True)
        return task_folder / "messages.json"
    
    def save_message(self, task_id: str, message: Message) -> bool:
        """
        保存单条消息
        
        Args:
            task_id: 任务ID
            message: 消息对象
            
        Returns:
            是否保存成功
        """
        try:
            message_file = self._get_message_file_path(task_id)
            
            # 读取现有消息
            if message_file.exists():
                with open(message_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    batch = MessageBatch(**data)
            else:
                batch = MessageBatch(task_id=task_id)
            
            # 添加新消息
            batch.messages.append(message)
            batch.updated_at = datetime.now()
            
            # 保存
            with open(message_file, 'w', encoding='utf-8') as f:
                json.dump(batch.model_dump(), f, ensure_ascii=False, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving message: {e}")
            return False
    
    def save_messages(self, task_id: str, messages: List[Message]) -> bool:
        """
        批量保存消息
        
        Args:
            task_id: 任务ID
            messages: 消息列表
            
        Returns:
            是否保存成功
        """
        try:
            message_file = self._get_message_file_path(task_id)
            
            # 读取现有消息
            if message_file.exists():
                with open(message_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    batch = MessageBatch(**data)
            else:
                batch = MessageBatch(task_id=task_id)
            
            # 添加新消息
            batch.messages.extend(messages)
            batch.updated_at = datetime.now()
            
            # 保存
            with open(message_file, 'w', encoding='utf-8') as f:
                json.dump(batch.model_dump(), f, ensure_ascii=False, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving messages: {e}")
            return False
    
    def get_messages_by_team(self, task_id: str, team_id: str) -> List[Message]:
        """
        获取某个团队的所有消息
        
        Args:
            task_id: 任务ID
            team_id: 团队ID
            
        Returns:
            消息列表
        """
        try:
            message_file = self._get_message_file_path(task_id)
            if not message_file.exists():
                return []
            
            with open(message_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batch = MessageBatch(**data)
            
            # 筛选出该团队的消息
            return [msg for msg in batch.messages if msg.team_id == team_id]
        except Exception as e:
            print(f"Error getting messages by team: {e}")
            return []
    
    def get_messages_by_agent(self, task_id: str, agent_id: str) -> List[Message]:
        """
        获取某个智能体可见的所有消息
        智能体可以看到所有包含自己的团队的消息
        
        Args:
            task_id: 任务ID
            agent_id: 智能体ID
            
        Returns:
            消息列表
        """
        try:
            message_file = self._get_message_file_path(task_id)
            if not message_file.exists():
                return []
            
            with open(message_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batch = MessageBatch(**data)
            
            # 筛选出智能体所在团队的消息
            visible_messages = []
            for msg in batch.messages:
                # team_id是一个由agent_id组成的字符串
                if agent_id in msg.team_id:
                    visible_messages.append(msg)
            
            return visible_messages
        except Exception as e:
            print(f"Error getting messages by agent: {e}")
            return []
    
    def get_messages_by_round(self, task_id: str, round_stage: RoundStage) -> List[Message]:
        """
        获取某个轮次的所有消息
        
        Args:
            task_id: 任务ID
            round_stage: 轮次阶段
            
        Returns:
            消息列表
        """
        try:
            message_file = self._get_message_file_path(task_id)
            if not message_file.exists():
                return []
            
            with open(message_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batch = MessageBatch(**data)
            
            return [msg for msg in batch.messages if msg.round_stage == round_stage]
        except Exception as e:
            print(f"Error getting messages by round: {e}")
            return []
    
    def get_all_messages(self, task_id: str) -> List[Message]:
        """
        获取任务的所有消息
        
        Args:
            task_id: 任务ID
            
        Returns:
            消息列表
        """
        try:
            message_file = self._get_message_file_path(task_id)
            if not message_file.exists():
                return []
            
            with open(message_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batch = MessageBatch(**data)
            
            return batch.messages
        except Exception as e:
            print(f"Error getting all messages: {e}")
            return []
    
    def get_latest_message(self, task_id: str, team_id: Optional[str] = None) -> Optional[Message]:
        """
        获取最新消息
        
        Args:
            task_id: 任务ID
            team_id: 团队ID(可选)
            
        Returns:
            最新消息
        """
        if team_id:
            messages = self.get_messages_by_team(task_id, team_id)
        else:
            messages = self.get_all_messages(task_id)
        
        return messages[-1] if messages else None
    
    def get_message_count(self, task_id: str, team_id: Optional[str] = None) -> int:
        """
        获取消息数量
        
        Args:
            task_id: 任务ID
            team_id: 团队ID(可选)
            
        Returns:
            消息数量
        """
        if team_id:
            return len(self.get_messages_by_team(task_id, team_id))
        else:
            return len(self.get_all_messages(task_id))
