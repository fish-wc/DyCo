from pathlib import Path

from configs import ConfigLoader
from src.models.config import AgentConfig, SystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_system_configuration_loads():
    loader = ConfigLoader(PROJECT_ROOT / "configs")

    config = loader.load_system_config()

    assert isinstance(config, SystemConfig)
    assert config.model.model_name == "gpt-4o-mini-2024-07-18"
    assert config.model.temperature == 0.7
    assert loader.get_agent_ids() == [
        "entj_001",
        "intj_001",
        "estp_001",
        "isfj_001",
    ]


def test_all_agent_configs_have_valid_models():
    loader = ConfigLoader(PROJECT_ROOT / "configs")
    agent_files = sorted((PROJECT_ROOT / "configs" / "agents").glob("*.yaml"))

    assert len(agent_files) == 16
    for agent_file in agent_files:
        config = loader.load_agent_config(agent_file.stem)
        assert isinstance(config, AgentConfig)
        assert config.agent_id == agent_file.stem
        assert len(config.mbti_type) == 4
        assert config.personality.mbti_type == config.mbti_type


def test_legacy_load_config_dispatches_by_agent_id():
    loader = ConfigLoader(PROJECT_ROOT / "configs")

    assert isinstance(loader.load_config(), SystemConfig)
    assert isinstance(loader.load_config(agent_id="entj_001"), AgentConfig)
