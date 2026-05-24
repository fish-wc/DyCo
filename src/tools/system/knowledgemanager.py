'''
进行知识库管理的工具模块,包括知识的存储、检索和更新等功能。
将会被集成到base_agent.py的system工具中,供smolagent自由调用。

使用 FAISS 向量数据库进行存储管理,采用 embedding 模型进行文本嵌入。
采用统一的共享知识库设计,支持相似度阈值筛选。
'''

import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import uuid
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from src.utils.llm_client import tool

class VectorKnowledgeBase:
    """基于FAISS的向量知识库"""
    
    def __init__(self, db_path: Path, embedding_client, embedding_model: str, 
                 dimension: Optional[int] = None, logger: Optional[logging.Logger] = None):
        """
        初始化向量知识库
        
        Args:
            db_path: 数据库路径
            embedding_client: embedding 客户端
            embedding_model: embedding 模型名称
            dimension: 向量维度 (如果为None则自动检测)
            logger: 日志记录器
        """
        if not FAISS_AVAILABLE:
            raise ImportError("需要安装 faiss 库: pip install faiss-cpu 或 faiss-gpu")
        
        self.db_path = db_path
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.logger = logger or logging.getLogger(__name__)
        
        # 自动检测向量维度
        if dimension is None:
            self.logger.info("自动检测向量维度...")
            self.dimension = self._detect_dimension()
            self.logger.info(f"检测到向量维度: {self.dimension}")
        else:
            self.dimension = dimension
        
        # 确保目录存在
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 索引文件和元数据文件
        self.index_file = self.db_path / "faiss.index"
        self.metadata_file = self.db_path / "metadata.json"
        
        # 加载或创建索引
        self.index = self._load_or_create_index()
        self.metadata: List[Dict[str, Any]] = self._load_metadata()
    
    def _detect_dimension(self) -> int:
        """自动检测向量维度"""
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=["测试文本用于检测向量维度"]
            )
            dimension = len(response.data[0].embedding)
            return dimension
        except Exception as e:
            self.logger.warning(f"自动检测维度失败: {e}, 使用默认值 2048")
            return 2048  # 智谱 embedding-3 默认维度
    
    def _load_or_create_index(self) -> Any:
        """加载或创建 FAISS 索引"""
        if self.index_file.exists():
            try:
                index = faiss.read_index(str(self.index_file))
                # 验证维度是否匹配
                if index.d != self.dimension:
                    self.logger.warning(
                        f"索引维度({index.d})与当前维度({self.dimension})不匹配，重建索引"
                    )
                    return faiss.IndexFlatL2(self.dimension)
                return index
            except Exception as e:
                self.logger.warning(f"加载索引失败: {e}, 创建新索引")
        
        # 创建新的 L2 距离索引
        return faiss.IndexFlatL2(self.dimension)
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"加载元数据失败: {e}, 返回空列表")
        return []
    
    def _save_index(self):
        """保存 FAISS 索引"""
        faiss.write_index(self.index, str(self.index_file))
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2, default=str)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示 (numpy数组)
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            
            # 记录 embedding token 消耗
            try:
                from src.variables.tokens import record_token_usage
                if hasattr(response, 'usage'):
                    usage = response.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'total_tokens', 0)
                    record_token_usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,  # embedding 没有输出token
                        label=f"embedding: {text[:30]}...",
                        category="embedding"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"记录embedding token失败: {e}")
            
            # 提取向量
            vector = response.data[0].embedding
            vec_array = np.array(vector, dtype=np.float32)
            
            # 验证维度
            if len(vec_array) != self.dimension:
                self.logger.error(
                    f"向量维度不匹配: 期望{self.dimension}, 实际{len(vec_array)}"
                )
                return np.zeros(self.dimension, dtype=np.float32)
            
            return vec_array
        except Exception as e:
            self.logger.error(f"获取embedding失败: {e}")
            # 返回零向量
            return np.zeros(self.dimension, dtype=np.float32)
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量获取文本的向量表示（提高效率）
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        try:
            # 批量调用 embedding API
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            # 记录 embedding token 消耗
            try:
                from src.variables.tokens import record_token_usage
                if hasattr(response, 'usage'):
                    usage = response.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'total_tokens', 0)
                    record_token_usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        label=f"batch_embedding: {len(texts)} texts",
                        category="embedding"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"记录embedding token失败: {e}")
            
            # 提取所有向量
            vectors = []
            for i, item in enumerate(response.data):
                vector = item.embedding
                vec_array = np.array(vector, dtype=np.float32)
                
                # 验证维度
                if len(vec_array) != self.dimension:
                    self.logger.error(
                        f"向量{i}维度不匹配: 期望{self.dimension}, 实际{len(vec_array)}"
                    )
                    vec_array = np.zeros(self.dimension, dtype=np.float32)
                
                vectors.append(vec_array)
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"批量获取embedding失败: {e}")
            # 返回零向量列表
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
    
    def add(self, knowledge: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加知识到向量库
        
        Args:
            knowledge: 知识内容
            metadata: 额外的元数据
            
        Returns:
            知识ID
        """
        # 生成知识ID
        knowledge_id = str(uuid.uuid4())
        
        # 获取向量表示
        vector = self._get_embedding(knowledge)
        
        # 添加到 FAISS 索引
        # FAISS 需要 2D 数组 (n_samples, dimension)
        vector_2d = vector.reshape(1, -1)
        self.index.add(vector_2d)
        
        # 保存元数据
        entry = {
            'id': knowledge_id,
            'content': knowledge,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'index_position': len(self.metadata)  # 在FAISS索引中的位置
        }
        self.metadata.append(entry)
        
        # 持久化
        self._save_index()
        self._save_metadata()
        
        return knowledge_id
    
    def add_batch(self, knowledge_list: List[Dict[str, Any]]) -> List[str]:
        """
        批量添加知识到向量库（提高效率）
        
        Args:
            knowledge_list: 知识列表，每项包含 {'knowledge': str, 'metadata': dict}
            
        Returns:
            知识ID列表
        """
        if not knowledge_list:
            return []
        
        # 提取所有知识内容
        texts = [item['knowledge'] for item in knowledge_list]
        
        # 批量获取向量
        self.logger.info(f"批量获取 {len(texts)} 个文本的embedding...")
        vectors = self._get_embeddings_batch(texts)
        
        # 批量添加到索引
        knowledge_ids = []
        entries_to_add = []
        
        for i, (text, vector, item) in enumerate(zip(texts, vectors, knowledge_list)):
            # 生成知识ID
            knowledge_id = str(uuid.uuid4())
            knowledge_ids.append(knowledge_id)
            
            # 准备元数据条目
            entry = {
                'id': knowledge_id,
                'content': text,
                'metadata': item.get('metadata', {}),
                'created_at': datetime.now().isoformat(),
                'index_position': len(self.metadata) + i
            }
            entries_to_add.append(entry)
        
        # 一次性添加所有向量到FAISS索引
        if vectors:
            vectors_2d = np.vstack(vectors)  # 转换为2D数组 (n_samples, dimension)
            self.index.add(vectors_2d)
        
        # 一次性更新所有元数据
        self.metadata.extend(entries_to_add)
        
        # 一次性持久化（只保存一次，大幅提升效率）
        self._save_index()
        self._save_metadata()
        
        self.logger.info(f"批量添加完成: {len(knowledge_ids)} 个知识点")
        
        return knowledge_ids
    
    def search(self, query: str, top_k: int = 5, distance_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        搜索相似知识
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
            distance_threshold: 距离阈值,超过此阈值的结果将被过滤(L2距离,越小越相似)
                               默认None表示不过滤。推荐值: 0.5-1.5 (根据embedding模型调整)
            
        Returns:
            相似知识列表 (包含距离分数)
        """
        if self.index.ntotal == 0:
            return []
        
        # 获取查询向量
        query_vector = self._get_embedding(query)
        query_vector_2d = query_vector.reshape(1, -1)
        
        # 搜索
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector_2d, k)
        
        # 构建结果并应用距离阈值筛选
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # 应用距离阈值筛选
            if distance_threshold is not None and dist > distance_threshold:
                continue
            
            if idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry['distance'] = float(dist)
                entry['rank'] = i + 1
                results.append(entry)
        
        return results
    
    def delete_by_content(self, knowledge: str) -> bool:
        """
        通过内容删除知识 (精确匹配)
        
        Args:
            knowledge: 知识内容
            
        Returns:
            是否删除成功
        """
        # 找到匹配的条目
        matched_indices = []
        for i, entry in enumerate(self.metadata):
            if entry['content'] == knowledge:
                matched_indices.append(i)
        
        if not matched_indices:
            return False
        
        # FAISS 不支持直接删除,需要重建索引
        # 保留未删除的条目
        new_metadata = [entry for i, entry in enumerate(self.metadata) if i not in matched_indices]
        
        # 重建索引
        self._rebuild_index(new_metadata)
        
        return True
    
    def update_by_content(self, old_knowledge: str, new_knowledge: str, 
                         similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        通过相似度搜索更新知识。只更新相似度最高且超过阈值的一条知识。
        
        Args:
            old_knowledge: 旧知识内容（用于搜索）
            new_knowledge: 新知识内容
            similarity_threshold: 相似度阈值（L2距离，越小越相似）。
                                只有距离小于此阈值的知识才会被更新。
                                推荐值: 0.5（严格）, 0.8（适中）, 1.2（宽松）
            
        Returns:
            更新结果字典 {"success": bool, "distance": float, "threshold": float}
        """
        if self.index.ntotal == 0:
            return {
                "success": False,
                "distance": None,
                "threshold": similarity_threshold
            }
        
        # 使用向量搜索找到最相似的知识
        search_results = self.search(old_knowledge, top_k=1, distance_threshold=similarity_threshold)
        
        if len(search_results) == 0:
            # 没有找到相似度足够高的知识
            self.logger.warning(
                f"未找到相似度足够高的知识进行更新 (阈值: {similarity_threshold})"
            )
            return {
                "success": False,
                "distance": None,
                "threshold": similarity_threshold
            }
        
        # 取最相似的那条
        most_similar = search_results[0]
        matched_index = most_similar['index_position']
        distance = most_similar['distance']
        
        self.logger.info(
            f"找到匹配知识 (距离: {distance:.4f}): {most_similar['content'][:50]}..."
        )
        
        # 更新元数据
        self.metadata[matched_index]['content'] = new_knowledge
        self.metadata[matched_index]['updated_at'] = datetime.now().isoformat()
        
        # 重建索引 (因为向量改变了)
        self._rebuild_index(self.metadata)
        
        return {
            "success": True,
            "distance": float(distance),
            "threshold": similarity_threshold
        }
    
    def _rebuild_index(self, new_metadata: List[Dict[str, Any]]):
        """
        重建 FAISS 索引
        
        Args:
            new_metadata: 新的元数据列表
        """
        # 创建新索引
        new_index = faiss.IndexFlatL2(self.dimension)
        
        # 重新添加所有向量
        for entry in new_metadata:
            vector = self._get_embedding(entry['content'])
            vector_2d = vector.reshape(1, -1)
            new_index.add(vector_2d)
        
        # 更新索引位置
        for i, entry in enumerate(new_metadata):
            entry['index_position'] = i
        
        # 替换旧索引和元数据
        self.index = new_index
        self.metadata = new_metadata
        
        # 持久化
        self._save_index()
        self._save_metadata()


class KnowledgeManager:
    """知识库管理器 - 管理统一的共享知识库"""
    
    def __init__(self, 
                 workspace_root: str,
                 embedding_client,
                 embedding_model: str,
                 dimension: Optional[int] = None,
                 distance_threshold: float = 1.0,
                 knowledge_base_id: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化知识库管理器
        
        Args:
            workspace_root: 工作空间根目录
            embedding_client: embedding 客户端
            embedding_model: embedding 模型名称
            dimension: 向量维度 (如果为None则自动检测)
            distance_threshold: 默认距离阈值,用于筛选低相似度结果 (默认1.0)
            knowledge_base_id: 知识库标识符(如task_id)。如果提供，将在workspace/knowledge/{knowledge_base_id}/下创建独立知识库；
                              如果为None，则使用默认的shared共享知识库
            logger: 日志记录器
        """
        self.workspace_root = Path(workspace_root)
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.distance_threshold = distance_threshold
        self.knowledge_base_id = knowledge_base_id
        
        # 设置日志记录器
        if logger is None:
            raise ValueError("必须提供 logger 参数")
        self.logger = logger
        
        # 构建知识库路径
        if knowledge_base_id is not None and knowledge_base_id.strip():
            # 使用独立的知识库目录(基于task_id等标识符)
            self.knowledge_path = self.workspace_root / "knowledge" / knowledge_base_id
            self.logger.info(f"初始化独立知识库 (ID: {knowledge_base_id})...")
            self.logger.debug(f"知识库路径: {self.knowledge_path}")
        else:
            # 使用默认的共享知识库
            self.knowledge_path = self.workspace_root / "knowledge" / "shared"
            self.logger.info("初始化共享知识库...")
            self.logger.debug(f"知识库路径: {self.knowledge_path}")
        
        # 初始化向量知识库
        self.knowledge_base = VectorKnowledgeBase(
            self.knowledge_path,
            self.embedding_client,
            self.embedding_model,
            self.dimension,
            self.logger
        )
        
        kb_type = f"独立知识库(ID: {knowledge_base_id})" if knowledge_base_id else "共享知识库"
        self.logger.info(
            f"知识库管理器初始化完成 - {kb_type} "
            f"(维度: {self.knowledge_base.dimension}, 阈值: {self.distance_threshold}, "
            f"路径: {self.knowledge_path})"
        )
    
    def add_knowledge(self, 
                     knowledge: str,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        添加知识到共享知识库
        
        Args:
            knowledge: 知识内容
            metadata: 额外的元数据
            
        Returns:
            操作结果 {"success": bool, "message": str, "knowledge_id": str}
        """
        if not knowledge or not knowledge.strip():
            return {
                "success": False,
                "message": "知识内容不能为空",
                "knowledge_id": None
            }
        
        try:
            knowledge_id = self.knowledge_base.add(knowledge, metadata)
            self.logger.info(f"知识已添加到共享库: {knowledge_id}")
            
            return {
                "success": True,
                "message": "知识添加成功",
                "knowledge_id": knowledge_id
            }
            
        except Exception as e:
            self.logger.error(f"添加知识失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"添加知识失败: {str(e)}",
                "knowledge_id": None
            }
    
    def add_knowledge_batch(self, 
                           knowledge_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量添加知识到共享知识库（提高效率）
        
        Args:
            knowledge_list: 知识列表，每项包含 {'knowledge': str, 'metadata': dict}
            
        Returns:
            操作结果 {"success": bool, "message": str, "knowledge_ids": List[str], "count": int}
        """
        if not knowledge_list:
            return {
                "success": False,
                "message": "知识列表不能为空",
                "knowledge_ids": [],
                "count": 0
            }
        
        # 过滤空知识
        valid_items = [
            item for item in knowledge_list 
            if item.get('knowledge') and item['knowledge'].strip()
        ]
        
        if not valid_items:
            return {
                "success": False,
                "message": "没有有效的知识内容",
                "knowledge_ids": [],
                "count": 0
            }
        
        try:
            knowledge_ids = self.knowledge_base.add_batch(valid_items)
            self.logger.info(f"批量添加知识到共享库: {len(knowledge_ids)} 个")
            
            return {
                "success": True,
                "message": f"成功添加 {len(knowledge_ids)} 个知识点",
                "knowledge_ids": knowledge_ids,
                "count": len(knowledge_ids)
            }
            
        except Exception as e:
            self.logger.error(f"批量添加知识失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"批量添加知识失败: {str(e)}",
                "knowledge_ids": [],
                "count": 0
            }
    
    def delete_knowledge(self, knowledge: str) -> Dict[str, Any]:
        """
        从共享知识库删除知识 (精确匹配)
        
        Args:
            knowledge: 知识内容
            
        Returns:
            操作结果 {"success": bool, "message": str}
        """
        if not knowledge or not knowledge.strip():
            return {
                "success": False,
                "message": "知识内容不能为空"
            }
        
        try:
            if self.knowledge_base.delete_by_content(knowledge):
                self.logger.info("从共享库删除知识")
                return {
                    "success": True,
                    "message": "知识删除成功"
                }
            else:
                return {
                    "success": False,
                    "message": "未找到匹配的知识"
                }
                
        except Exception as e:
            self.logger.error(f"删除知识失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"删除知识失败: {str(e)}"
            }
    
    def query_knowledge(self,
                       query: str,
                       top_k: int = 5,
                       distance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        从共享知识库查询相关知识，自动筛选低相似度结果
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            distance_threshold: 距离阈值,超过此阈值的结果将被过滤。
                              如果为None,则使用初始化时设置的默认阈值
            
        Returns:
            查询结果 {"success": bool, "message": str, "results": [...], "filtered_count": int}
        """
        if not query or not query.strip():
            return {
                "success": False,
                "message": "查询内容不能为空",
                "results": [],
                "filtered_count": 0
            }
        
        try:
            # 使用提供的阈值或默认阈值
            threshold = distance_threshold if distance_threshold is not None else self.distance_threshold
            
            # 查询知识库(应用距离阈值筛选)
            results = self.knowledge_base.search(query, top_k, distance_threshold=threshold)
            
            filtered_count = top_k - len(results) if len(results) < top_k else 0
            
            if len(results) == 0:
                self.logger.info(f"查询完成: 未找到相似度足够高的知识 (阈值: {threshold})")
                return {
                    "success": True,
                    "message": f"未找到相关知识 (相似度阈值: {threshold})",
                    "results": [],
                    "filtered_count": filtered_count
                }
            
            self.logger.info(f"查询完成: 找到{len(results)}条知识 (过滤{filtered_count}条低相似度结果)")
            
            return {
                "success": True,
                "message": f"查询成功，找到{len(results)}条相关知识",
                "results": results,
                "filtered_count": filtered_count
            }
            
        except Exception as e:
            self.logger.error(f"查询知识失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"查询知识失败: {str(e)}",
                "results": [],
                "filtered_count": 0
            }
    
    def update_knowledge(self,
                        old_knowledge: str,
                        new_knowledge: str,
                        similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        更新共享知识库中的知识（基于相似度搜索）
        
        Args:
            old_knowledge: 旧知识内容（用于搜索最相似的知识）
            new_knowledge: 新知识内容
            similarity_threshold: 相似度阈值（L2距离）。只更新距离小于此阈值的知识。
                                默认0.8（适中）
            
        Returns:
            操作结果 {"success": bool, "message": str}
        """
        if not old_knowledge or not old_knowledge.strip():
            return {
                "success": False,
                "message": "旧知识内容不能为空"
            }
        
        if not new_knowledge or not new_knowledge.strip():
            return {
                "success": False,
                "message": "新知识内容不能为空"
            }
        
        try:
            result = self.knowledge_base.update_by_content(old_knowledge, new_knowledge, similarity_threshold)
            
            if result["success"]:
                self.logger.info("共享库知识已更新")
                return {
                    "success": True,
                    "message": "知识更新成功",
                    "distance": result["distance"],
                    "threshold": result["threshold"]
                }
            else:
                return {
                    "success": False,
                    "message": f"未找到相似度足够高的知识 (阈值: {similarity_threshold})",
                    "distance": result["distance"],
                    "threshold": result["threshold"]
                }
                
        except Exception as e:
            self.logger.error(f"更新知识失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"更新知识失败: {str(e)}"
            }


# ============= Smolagent 工具封装 =============

# 全局知识库管理器实例 (需要在使用前初始化)
_global_knowledge_manager: Optional[KnowledgeManager] = None

# NOTE 高层调用的时候初始化。
def init_knowledge_manager(workspace_root: str,
                           embedding_client,
                           embedding_model: str,
                           dimension: Optional[int] = None,
                           distance_threshold: float = 1.0,
                           knowledge_base_id: Optional[str] = None,
                           logger: Optional[logging.Logger] = None):
    """
    初始化全局知识库管理器
    
    Args:
        workspace_root: 工作空间根目录
        embedding_client: embedding 客户端
        embedding_model: embedding 模型名称
        dimension: 向量维度 (如果为None则自动检测)
        distance_threshold: 默认距离阈值,用于筛选低相似度结果 (默认1.0)
        knowledge_base_id: 知识库标识符(如task_id)。如果提供，将创建独立的知识库；如果为None，则使用共享知识库
        logger: 日志记录器
    
    Example:
        # 使用共享知识库
        init_knowledge_manager(workspace_root="workspace", embedding_client=client, 
                              embedding_model="text-embedding-3-small", logger=logger)
        
        # 使用独立知识库(基于task_id)
        init_knowledge_manager(workspace_root="workspace", embedding_client=client,
                              embedding_model="text-embedding-3-small", 
                              knowledge_base_id="test_task_8agents1", logger=logger)
    """
    global _global_knowledge_manager
    
    if logger:
        if knowledge_base_id:
            logger.info(f"正在初始化知识库管理器 (知识库ID: {knowledge_base_id})...")
        else:
            logger.info("正在初始化知识库管理器 (共享模式)...")
    
    _global_knowledge_manager = KnowledgeManager(
        workspace_root=workspace_root,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        dimension=dimension,
        distance_threshold=distance_threshold,
        knowledge_base_id=knowledge_base_id,
        logger=logger
    )
    
    if logger:
        logger.info("知识库管理器初始化成功")


@tool
def add_knowledge_tool(knowledge: str, metadata: dict = None) -> str:
    """
    添加知识到共享知识库。
    
    改进版本：支持分离的 content 和 metadata
    - knowledge: 纯净的知识内容（用于向量嵌入）
    - metadata: 元数据信息（作为附加信息存储，不参与嵌入）
    
    这样可以避免 XML 标签稀释语义密度，提高检索准确度。
    
    Args:
        knowledge: 需要添加的知识内容（纯文本，不含 XML 标签）
        metadata: 可选的元数据字典（如 agent_id, mbti_type 等）
    
    Returns:
        操作结果的JSON字符串，包含成功状态和知识ID
    """
    if _global_knowledge_manager is None:
        return json.dumps({"success": False, "message": "知识库管理器未初始化"}, ensure_ascii=False)
    
    result = _global_knowledge_manager.add_knowledge(knowledge=knowledge, metadata=metadata)
    return json.dumps(result, ensure_ascii=False)


@tool
def add_knowledge_batch_tool(knowledge_list: list) -> str:
    """
    批量添加知识到共享知识库（提高效率）。
    
    批量处理可以显著提升性能，因为：
    1. 一次API调用获取多个文本的embedding
    2. 一次性保存所有向量和元数据
    
    Args:
        knowledge_list: 知识列表，每项包含 {'knowledge': str, 'metadata': dict}
                       例如: [
                           {'knowledge': '知识内容1', 'metadata': {'type': 'evaluation'}},
                           {'knowledge': '知识内容2', 'metadata': {'type': 'decision'}}
                       ]
    
    Returns:
        操作结果的JSON字符串，包含成功状态、知识ID列表和添加数量
    """
    if _global_knowledge_manager is None:
        return json.dumps({"success": False, "message": "知识库管理器未初始化"}, ensure_ascii=False)
    
    result = _global_knowledge_manager.add_knowledge_batch(knowledge_list=knowledge_list)
    return json.dumps(result, ensure_ascii=False)


@tool
def delete_knowledge_tool(knowledge: str) -> str:
    """
    从共享知识库删除知识。如果你觉得某条知识不再适用或需要移除，可以调用此工具进行删除。
    注意：需要删除的知识knowledge参数需要与知识库中已有的知识完全匹配才能进行删除。
    
    Args:
        knowledge: 要删除的知识内容（需精确匹配）
    
    Returns:
        操作结果的JSON字符串
    """
    if _global_knowledge_manager is None:
        return json.dumps({"success": False, "message": "知识库管理器未初始化"}, ensure_ascii=False)
    
    result = _global_knowledge_manager.delete_knowledge(knowledge=knowledge)
    return json.dumps(result, ensure_ascii=False)


@tool
def query_knowledge_tool(query: str, top_k: int = 5,distance_threshold : float = None) -> str:
    """
    从共享知识库查询相关知识。系统会自动过滤相似度过低的结果，只返回真正相关的知识。
    如果你在解决问题时需要思考曾经的经验和知识，可以调用此工具来获取相关信息。
    
    Args:
        query: 查询文本
        top_k: 返回的最大结果数 (默认5)
    
    Returns:
        查询结果的JSON字符串。包含results列表（每项包含content、distance、rank等字段）
        和filtered_count（被过滤的低相似度结果数量）。distance越小表示越相似。
    """
    if _global_knowledge_manager is None:
        return json.dumps({"success": False, "message": "知识库管理器未初始化"}, ensure_ascii=False)
    
    result = _global_knowledge_manager.query_knowledge(
        query=query,
        top_k=top_k,
        distance_threshold=distance_threshold  # 使用较宽松的阈值，确保返回足够结果
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def update_knowledge_tool(old_knowledge: str, new_knowledge: str) -> str:
    """
    更新共享知识库中的知识。系统会搜索与old_knowledge最相似的知识，
    如果相似度足够高（距离<0.8），则更新为new_knowledge。
    注意：这是基于向量相似度的智能更新，不需要精确匹配。
    
    Args:
        old_knowledge: 旧知识的描述（用于搜索）
        new_knowledge: 新的知识内容
    
    Returns:
        操作结果的JSON字符串，包含success、message、distance（最相似知识的距离）和threshold（阈值）
    """
    if _global_knowledge_manager is None:
        return json.dumps({"success": False, "message": "知识库管理器未初始化"}, ensure_ascii=False)
    
    result = _global_knowledge_manager.update_knowledge(
        old_knowledge=old_knowledge,
        new_knowledge=new_knowledge,
        similarity_threshold=0.3  # 使用适中的阈值
    )
    return json.dumps(result, ensure_ascii=False)

