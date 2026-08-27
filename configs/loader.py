"""
配置加载器：从 configs/ 目录读取 system.yaml 与各 agent 的 YAML 配置。

约定：
- system.yaml 描述系统级配置（工作空间、模型、embedding、agent 列表等）。
- agents/<agent_id>.yaml 描述单个 agent（角色先验、模型、知识库路径等）。
- ModelConfig 中的 `api_key` 与 `model_url` 字段填写的是**环境变量名**，
  运行期由 src/utils/llm_client.py 通过 os.getenv() 解析为真实值，
  因此密钥永远不会进入仓库。
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import yaml

from src.models.config import AgentConfig, ModelConfig, SystemConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """从 configs/ 目录加载系统配置与 agent 配置。"""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None) -> None:
        if config_dir is None:
            # 默认指向项目根目录下的 configs/
            project_root = Path(__file__).resolve().parents[1]
            config_dir = project_root / "configs"
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")

    # ------------------------------------------------------------------ #
    # 系统配置
    # ------------------------------------------------------------------ #
    def _read_yaml(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _build_model_config(self, raw: dict) -> ModelConfig:
        """把 YAML 中的模型配置块转换为 ModelConfig。"""
        return ModelConfig(
            model_name=raw["model_name"],
            model_url=raw["model_url"],
            api_key=raw["api_key"],
            temperature=raw.get("temperature", 0.7),
            max_tokens=raw.get("max_tokens"),
            timeout=raw.get("timeout", 60),
            max_steps=raw.get("max_steps", 10),
            verbosity_level=raw.get("verbosity_level", 0),
            dimension=raw.get("dimension"),
        )

    def load_system_config(self) -> SystemConfig:
        """加载 system.yaml 并返回 SystemConfig。"""
        raw = self._read_yaml(self.config_dir / "system.yaml")
        system_config = SystemConfig(
            workspace_root=raw.get("workspace_root", "workspace"),
            message_storage_type=raw.get("message_storage_type", "json"),
            log_level=raw.get("log_level", "INFO"),
            max_concurrent_agents=raw.get("max_concurrent_agents", 4),
            max_evaluation_agents=raw.get("max_evaluation_agents", 4),
            max_rounds=raw.get("max_rounds", 10),
            max_final_discussion_rounds=raw.get("max_final_discussion_rounds", 15),
            collate_memory_window=raw.get("collate_memory_window", 10),
            attitude_memory_window=raw.get("attitude_memory_window", 2),
            web_search_count=raw.get("web_search_count", 2),
            num_listeners_sampled_per_round=raw.get("num_listeners_sampled_per_round", 1),
            agents=raw.get("agents", []),
            model=self._build_model_config(raw["model"]),
            embedding=self._build_model_config(raw["embedding"]),
            collate_model=(
                self._build_model_config(raw["collate_model"])
                if raw.get("collate_model")
                else None
            ),
        )
        return system_config

    # ------------------------------------------------------------------ #
    # agent 配置
    # ------------------------------------------------------------------ #
    def get_agent_ids(self) -> List[str]:
        """返回 system.yaml 中登记的 agent ID 列表（即默认组队）。"""
        raw = self._read_yaml(self.config_dir / "system.yaml")
        agent_entries = raw.get("agents", [])
        ids: List[str] = []
        for entry in agent_entries:
            if isinstance(entry, dict):
                # 形如 {agent_id: entj_001, config_file: agents/entj_001.yaml}
                agent_id = entry.get("agent_id")
                if agent_id is None:
                    raise ValueError(f"agents 列表条目缺少 agent_id 字段: {entry}")
                ids.append(str(agent_id))
            else:
                ids.append(str(entry))
        return ids

    def load_agent_config(self, agent_id: str) -> AgentConfig:
        """加载 agents/<agent_id>.yaml 并返回 AgentConfig。"""
        config_path = self.config_dir / "agents" / f"{agent_id}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"未找到 agent 配置文件: {config_path}")
        raw = self._read_yaml(config_path)

        model_raw = raw.get("model", {})
        # 允许 agent 级配置省略模型字段，回退到系统默认模型
        system_model = self.load_system_config().model
        model = ModelConfig(
            model_name=model_raw.get("model_name", system_model.model_name),
            model_url=model_raw.get("model_url", system_model.model_url),
            api_key=model_raw.get("api_key", system_model.api_key),
            temperature=model_raw.get("temperature", system_model.temperature),
            max_tokens=model_raw.get("max_tokens", system_model.max_tokens),
            timeout=model_raw.get("timeout", system_model.timeout),
            max_steps=model_raw.get("max_steps", system_model.max_steps),
            verbosity_level=model_raw.get("verbosity_level", system_model.verbosity_level),
            dimension=model_raw.get("dimension", system_model.dimension),
        )

        personality_raw = raw.get("personality", {})
        knowledge_raw = raw.get("knowledge", {})

        return AgentConfig(
            agent_id=raw["agent_id"],
            agent_name=raw["agent_name"],
            mbti_type=raw["mbti_type"],
            model=model,
            personality={
                "mbti_type": personality_raw.get("mbti_type", raw["mbti_type"]),
                "speaking_threshold": personality_raw.get("speaking_threshold", 6.0),
            },
            knowledge={
                "private_kb_path": knowledge_raw.get(
                    "private_kb_path", "workspace/knowledge/private"
                ),
                "shared_kb_path": knowledge_raw.get(
                    "shared_kb_path", "workspace/knowledge/shared"
                ),
                "vector_store_type": knowledge_raw.get("vector_store_type", "faiss"),
                "embedding_model": knowledge_raw.get("embedding_model"),
            },
            max_memory_items=raw.get("max_memory_items", 100),
            workspace_path=raw.get("workspace_path", "workspace"),
            memory_window_size=raw.get("memory_window_size", 6),
        )

    # ------------------------------------------------------------------ #
    # 兼容入口
    # ------------------------------------------------------------------ #
    def load_config(self, agent_id: Optional[str] = None):
        """
        兼容旧调用的统一入口：
        - load_config(agent_id=...) 返回该 agent 的 AgentConfig；
        - load_config()             返回 SystemConfig。
        """
        if agent_id is None:
            return self.load_system_config()
        return self.load_agent_config(agent_id)


# 模块级单例：`from configs import config_loader` 直接可用。
# 默认读取项目根目录下的 configs/；如需指定其他目录，请新建 ConfigLoader(path)。
config_loader = ConfigLoader()
