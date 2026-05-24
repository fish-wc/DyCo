"""
讨论摘要管理器
用于在讨论过程中收集和存储关键信息，供后续大纲生成使用
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class DiscussionSummaryManager:
    """
    讨论摘要管理器
    
    功能：
    1. 收集各动作的关键信息
    2. 持久化存储到JSON文件
    3. 读取摘要供大纲生成使用
    
    存储结构：
    {
        "task_id": "xxx",
        "task": "任务描述",
        "summaries": [
            {
                "timestamp": "2024-01-01 12:00:00",
                "action": "analyze_task",
                "agent_id": "enfj_001",
                "content": "关键信息"
            },
            ...
        ]
    }
    """
    
    def __init__(self, task_id: str, workspace_root: str = "workspace", model_name: str = "MBTI_MAS"):
        """
        初始化管理器
        
        Args:
            task_id: 任务ID
            workspace_root: 工作空间根目录
        """
        self.task_id = task_id
        self.workspace_root = Path(workspace_root)
        
        # 摘要文件路径：workspace/messages/<model_name>/<task_id>/discussion_summary.json
        self.summary_dir = self.workspace_root / "messages" / model_name / task_id
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.summary_dir / "discussion_summary.json"
        
        # 初始化数据结构
        self.data: Dict[str, Any] = {
            "task_id": task_id,
            "task": "",
            "summaries": []
        }
        
        # 如果文件已存在，加载
        if self.summary_file.exists():
            self._load()
    
    def set_task(self, task: str):
        """设置任务描述"""
        self.data["task"] = task
        self._save()
    
    def add_summary(self, action: str, agent_id: str, content: str, metadata: Optional[Dict] = None):
        """
        添加一条摘要
        
        Args:
            action: 动作名称（analyze_task, generate, attitude等）
            agent_id: 智能体ID
            content: 关键信息内容
            metadata: 额外元数据（可选）
        """
        summary_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "agent_id": agent_id,
            "content": content
        }
        
        if metadata:
            summary_item["metadata"] = metadata
        
        self.data["summaries"].append(summary_item)
        self._save()
    
    def add_search_queries(self, agent_id: str, search_queries: List[str]):
        """
        添加搜索查询（从generate的tool_calls中提取）
        
        Args:
            agent_id: 智能体ID
            search_queries: 搜索查询列表
        """
        if search_queries:
            content = " | ".join(search_queries)
            self.add_summary(
                action="web_search",
                agent_id=agent_id,
                content=content,
                metadata={"query_count": len(search_queries)}
            )
    
    def get_all_summaries(self) -> List[Dict]:
        """获取所有摘要"""
        return self.data.get("summaries", [])
    
    def get_summaries_by_action(self, action: str) -> List[Dict]:
        """获取指定动作的摘要"""
        return [s for s in self.data["summaries"] if s["action"] == action]
    
    def get_formatted_summary(self, max_items: int = 50) -> str:
        """
        获取格式化的摘要文本（用于提示词）
        
        Args:
            max_items: 最多返回的条目数
            
        Returns:
            格式化的摘要文本
        """
        summaries = self.data["summaries"][-max_items:]  # 取最近的N条
        
        if not summaries:
            return "暂无讨论记录"
        
        lines = []
        for idx, item in enumerate(summaries, 1):
            action = item["action"]
            agent_id = item["agent_id"]
            content = item["content"]
            
            # 格式化输出
            lines.append(f"{idx}. [{action}] {agent_id}: {content}")
        
        return "\n".join(lines)
    
    def get_summary_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        stats = {}
        for item in self.data["summaries"]:
            action = item["action"]
            stats[action] = stats.get(action, 0) + 1
        return stats
    
    def _load(self):
        """从文件加载数据"""
        try:
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception:
            # 加载失败，使用默认数据
            pass
    
    def _save(self):
        """保存数据到文件"""
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            # 保存失败，静默处理
            pass
    
    def clear(self):
        """清空所有摘要"""
        self.data["summaries"] = []
        self._save()
