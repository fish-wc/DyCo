"""
发言队列管理器
管理智能体的发言意愿和发言顺序
"""
from typing import List, Dict, Optional
from queue import PriorityQueue
from datetime import datetime
import threading

from ..models.message import SpeakingIntention

# 采用了优先队列来管理发言顺序,分数越高优先级越高
# 使用锁来保证线程安全
class SpeakingQueue:
    """发言队列管理器"""
    
    def __init__(self):
        """初始化发言队列"""
        # 使用优先队列,按照发言意愿分数排序
        self._queue = PriorityQueue()
        self._intentions: Dict[str, SpeakingIntention] = {}
        self._lock = threading.Lock()
        self._sequence_counter = 0
    
    def update_intention(self, agent_id: str, intention: SpeakingIntention):
        """
        更新智能体的发言意愿
        
        Args:
            agent_id: 智能体ID
            intention: 发言意愿对象
        """
        with self._lock:
            self._intentions[agent_id] = intention
            
            # 如果达到发言阈值,加入发言队列
            if intention.should_speak:
                # 优先队列按负分数排序(分数越高优先级越高)
                priority = -intention.intention_score
                self._queue.put((priority, datetime.now().timestamp(), agent_id))
    
    def get_next_speaker(self) -> Optional[str]:
        """
        获取下一个发言者
        
        Returns:
            下一个发言者的agent_id,如果队列为空返回None
        """
        with self._lock:
            if self._queue.empty():
                return None
            
            _, _, agent_id = self._queue.get()
            
            # 重置该智能体的发言意愿
            if agent_id in self._intentions:
                self._intentions[agent_id].intention_score = 0.0
            
            return agent_id
    
    def get_intention(self, agent_id: str) -> Optional[SpeakingIntention]:
        """
        获取智能体的发言意愿
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            发言意愿对象
        """
        return self._intentions.get(agent_id)
    
    def clear_queue(self):
        """清空发言队列"""
        with self._lock:
            while not self._queue.empty():
                self._queue.get()
            self._intentions.clear()
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()
    
    def get_all_intentions(self) -> Dict[str, SpeakingIntention]:
        """获取所有智能体的发言意愿"""
        return self._intentions.copy()
    
    def get_agents_ready_to_speak(self) -> List[str]:
        """获取所有达到发言阈值的智能体"""
        return [
            agent_id 
            for agent_id, intention in self._intentions.items() 
            if intention.should_speak
        ]
    
    def remove_agent(self, agent_id: str):
        """
        从队列中移除智能体
        
        Args:
            agent_id: 智能体ID
        """
        with self._lock:
            if agent_id in self._intentions:
                del self._intentions[agent_id]
    
    def get_next_sequence(self) -> int:
        """获取下一个发言序号"""
        with self._lock:
            self._sequence_counter += 1
            return self._sequence_counter
    
    def reset_sequence(self):
        """重置发言序号"""
        with self._lock:
            self._sequence_counter = 0
