# DyCo

[![论文](https://img.shields.io/badge/EMNLP%202026-Findings-b31b1b.svg)](PAPER_URL_TODO) [![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**DyCo（Dynamic Cognitive Role Coordination，动态认知角色协作）**是一个面向开放式、基于文献的多智能体研究综合与验证框架。它结合了结构化角色先验、动态组队、基于发言意愿的异步讨论，以及 **Exploration--Verification Alternation（EVA，探索—验证交替）**机制。

> **重要范围说明：** 本项目仅把 MBTI 作为可解释、可控的角色先验测试床。DyCo 不声称语言模型具有人的真实人格，也不对 MBTI 的心理测量学有效性作出主张。

**论文：** *DyCo: Dynamic Cognitive Role Coordination for Open-Ended Multi-Agent Research Synthesis and Verification*（EMNLP 2026 Findings；论文链接：`PAPER_URL_TODO`）。
**作者：** Zhiyu Zhang、Leheng Wu、Yijun Mo、Mingchu Zhong、Yulin Zhang、Chen Xi。

## 项目概览

![DyCo 架构图](figures/mbti_dmas_overview.png)

DyCo 将智能体组织为四类功能认知角色：

- **NT / Theorist（理论家）：** 抽象框架与战略分析；
- **NF / Innovator（创新者）：** 假设生成与模式发现；
- **ST / Verifier（验证者）：** 具体证据检查与批判性验证；
- **SF / Mediator（协调者）：** 综合、冲突解决与团队协调。

框架包含四个主要阶段：

1. **初始分析：** 智能体独立分析研究问题并提出方向；
2. **动态组队：** 根据亲和网络形成面向子任务的团队；
3. **意愿驱动讨论：** 智能体根据上下文中的发言意愿加入异步优先队列；
4. **代表协商：** 各团队代表整合发现并生成最终报告。

EVA 在探索和验证之间交替，使有潜力的假设在综合前反复接受质疑。

## 论文报告的结果

DeepResearch Bench 上的 RACE 分数（`均值 ± 标准差`，10 次独立运行；越高越好）：

| 方法 | Overall RACE |
| --- | ---: |
| SingleAgent | 37.80 ± 0.38 |
| ToTAgent | 40.62 ± 0.42 |
| MetaGPT | 39.72 ± 0.49 |
| AutoGen | 40.43 ± 0.47 |
| CrewAI | 40.88 ± 0.45 |
| 4×INFJ | 42.53 ± 0.30 |
| **DyCo-Diverse** | **42.75 ± 0.31** |

论文中的多样角色配置为 **ENTJ + INTJ + ESTP + ISFJ**。在报告设置下，该配置相对 CrewAI 提升 4.6%。匹配控制实验表明，大部分收益来自丰富且行为上具体的角色先验以及面向验证的协作机制，而不应简单归因于 MBTI 语义本身。DyCo-Diverse 与最强同质配置（4×INFJ）的差异在论文中没有达到统计显著。

论文还报告了 **DyCo-Efficient**：移除动态组队后，以约十三分之一的 token 成本保留大部分质量。在预算有限时建议优先使用该配置。

## 仓库状态与数据说明

本仓库包含核心框架、提示词、配置模板和评测入口。DeepResearch Bench 数据及生成的报告**不随本仓库再分发**，请从官方来源获取数据并遵守其许可证与使用条款。

正式发布前还需要补充以下信息：

- 论文 / ACL Anthology 链接：`PAPER_URL_TODO`；
- GitHub 仓库链接：`REPOSITORY_URL_TODO`；
- 数据集版本或 commit：`DATASET_VERSION_TODO`；
- 任务采样随机种子及完整运行 seed 列表：`SEEDS_TODO`；
- （如单独归档代码）代码 artifact DOI：`CODE_DOI_TODO`。

## 安装

要求 Python 3.10 或更高版本。`requirements.txt` 是精简运行依赖；原始开发环境的完整快照保存在 `requirements-full.txt`。

```bash
git clone REPOSITORY_URL_TODO
cd DyCo
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows PowerShell：.venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果当前平台无法安装 `faiss-cpu`，请单独安装与平台兼容的 FAISS 版本，再安装 `requirements.txt` 中的其他依赖。

## 配置

```bash
cp .env.example .env
```

Windows PowerShell 可使用 `Copy-Item .env.example .env`。请填写：

- `OPENAI_API_KEY`、`OPENAI_BASE_URL`：OpenAI 兼容的生成模型接口；
- `OPENAI_EMBEDDING_API_KEY`、`OPENAI_EMBEDDING_MODEL_URL`：embedding 接口；
- `JINA_API_KEY`：Jina 检索服务。

`configs/system.yaml` 中的 `api_key` 和 `model_url` 填的是**环境变量名**，不是密钥本身。不要提交 `.env` 或真实密钥。论文默认设置为 `gpt-4o-mini-2024-07-18`、temperature `0.7`、`text-embedding-3-small`，以及 Jina top-5 检索。可以使用其他 OpenAI 兼容服务，但结果将不能与论文直接比较。

## 快速开始：运行单个任务

下面的示例运行默认的 DyCo-Diverse 团队。它会调用外部模型、embedding 和检索服务，因此会产生服务费用。

```python
from src.workflows.mbtiagentsystem import MBTIAgentSystem

system = MBTIAgentSystem(
    config_dir="configs",
    agent_ids=["entj_001", "intj_001", "estp_001", "isfj_001"],
)
result = system.solve_task(
    "What are the applications and risks of large language models in medical diagnosis?"
)

if result["success"]:
    print(result["final_answer"])
else:
    raise RuntimeError(result.get("error_message", "DyCo 执行失败"))
```

运行期产物（消息、日志和任务级知识库）写入 `workspace/`，该目录已加入 Git 忽略列表。

## 测试

仓库包含不调用外部服务的配置 smoke test：

```bash
python -m pytest -q tests
```

如需检查完整本地代码树的语法：

```bash
python -m compileall -q src configs evaluate tests
```

## 复现论文评测生成

获取官方 benchmark 文件后，指定其中的 `query.jsonl`：

```bash
python evaluate/ours/generate_mbti_responses.py \\
  --query_file /path/to/query.jsonl \\
  --output_dir workspace/evaluation/dyco-diverse \\
  --agent_ids entj_001 intj_001 estp_001 isfj_001 \\
  --limit 1
```

去掉 `--limit 1` 可处理输入集合中的全部任务。使用 `--resume` 跳过已有元数据的任务，或使用 `--query_ids 1 15 16` 只处理指定任务。脚本会在输出目录生成 HTML、JSON 元数据和纯文本报告。

### 论文配置

`configs/system.yaml` 中的默认队伍是论文的 DyCo-Diverse：

```text
ENTJ + INTJ + ESTP + ISFJ
```

最强同质配置可以将同一个 ID 传入四次：

```bash
python evaluate/ours/generate_mbti_responses.py \\
  --query_file /path/to/query.jsonl \\
  --output_dir workspace/evaluation/4x-infj \\
  --agent_ids infj_001 infj_001 infj_001 infj_001
```

`configs/agents/` 下提供了全部 16 种角色配置。想修改默认队伍时，编辑 `configs/system.yaml` 末尾的 `agents` 列表。配置格式和论文设置的对应关系见 [configs/README.md](configs/README.md)。

## 项目结构

```text
configs/                 YAML 配置加载器与示例配置
src/agents/               Agent 基类与 16 种 MBTI 角色实现
src/communication/        消息、发言队列和团队管理
src/prompts/              角色和功能提示词模板
src/tools/                知识管理和网络检索工具
src/utils/                LLM 客户端、解析和讨论摘要
src/workflows/            四轮 DyCo 工作流与系统入口
evaluate/ours/            benchmark 回答生成脚本
figures/                  文档使用的发布图
```

## 实验细节

论文主实验使用 GPT-4o-mini（`gpt-4o-mini-2024-07-18`，temperature 0.7）、Jina web search（每个查询最多五个段落）、团队内最多八轮讨论，以及最多五轮代表层协商。DeepResearch Bench 覆盖 22 个领域；去重后得到 100 个任务，划分为 50 个开发任务和 50 个评测任务，论文主结果在留出的评测集上报告。除非另有说明，`n=10` 表示 10 次独立的全 benchmark 运行。

RACE 衡量 Comprehensiveness、Insight、Instruction Following 和 Readability，不直接衡量事实正确性或引用忠实度。因此论文同时报告了人工评测和小规模盲法事实性 / 来源忠实度审计。自动分数不应被解读为事实可靠性的保证。

## 局限与负责任使用

- 输出依赖于选用的模型、提示词、检索是否可用以及运行时的网页内容；
- 系统可能生成看似合理但没有证据支持的陈述，应对照原始来源核查；
- 不要将生成报告作为医疗、法律、金融、安全关键或其他高风险决策的唯一依据；
- 请遵守模型服务商、Jina 和 DeepResearch Bench 的条款，不要把私人或敏感数据发送给外部服务。

## 引用

如果使用 DyCo，请引用论文。官方 ACL Anthology 记录和仓库地址确定后，请同步替换 [CITATION.cff](CITATION.cff) 中的占位符。

```bibtex
@inproceedings{zhang2026dyco,
  title     = {DyCo: Dynamic Cognitive Role Coordination for Open-Ended Multi-Agent Research Synthesis and Verification},
  author    = {Zhang, Zhiyu and Wu, Leheng and Mo, Yijun and Zhong, Mingchu and Zhang, Yulin and Xi, Chen},
  booktitle = {Findings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {PAPER_URL_TODO},
  doi       = {PAPER_DOI_TODO}
}
```

## 许可证

源代码采用 [MIT License](LICENSE)。第三方数据集、API、模型输出和提示词衍生材料仍受其各自条款约束。

## 贡献

开发环境、问题报告、Pull Request 和可复现性要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
