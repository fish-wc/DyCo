# configs/

DyCo 的运行配置。克隆后无需修改即可跑通（前提是设置好 `.env` 中的 API 密钥）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `system.yaml` | 系统级配置：工作空间、讨论流程参数（轮数、检索条数等）、主模型 / embedding 模型、默认组队 |
| `agents/<type>_001.yaml` | 单个 agent 的配置：角色先验（MBTI 类型）、发言阈值、知识库路径 |
| `loader.py` | `ConfigLoader` 实现，把 YAML 解析为 `src/models/config.py` 中的 pydantic 结构 |

## 关键约定

1. **`api_key` / `model_url` 填的是环境变量名，不是真实值。**
   运行期由 `src/utils/llm_client.py` 通过 `os.getenv()` 解析，因此密钥不会进入仓库。
   请复制根目录 `.env.example` 为 `.env` 并填入真实密钥。
2. **agent_id 的前 4 个字符必须是 MBTI 类型**（如 `entj_001`），系统据此在
   `src/agents/agentsmanager.py` 的 `AGENT_CLASS_MAP` 中查找对应的 Agent 类。
3. **切换实验配置**只需改 `system.yaml` 末尾的 `agents` 列表：
   - DyCo-Diverse（论文主配置）：`entj_001, intj_001, estp_001, isfj_001`
   - 4×INFJ（最强同质基线）：把四个条目都换成 `infj_001`
4. 模型参数（temperature 等）在 `system.yaml` 的 `model` 块统一设置；
   单个 agent 需要覆盖时，取消其 YAML 中 `model` 块的注释即可。

## 与论文的对应

| 论文设置（附录 H） | 配置项 |
|---|---|
| GPT-4o-mini (`gpt-4o-mini-2024-07-18`), temperature 0.7 | `model.model_name` / `model.temperature` |
| JINA 检索，每查询 top-5 段落 | `web_search_count: 5` + `.env` 中 `JINA_API_KEY` |
| 团队内最大讨论轮数 $T_{\max}=8$ | `max_rounds: 8` |
| 代表层协商最大轮数 5 | `max_final_discussion_rounds: 5` |
