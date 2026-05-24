# DyCo

DyCo: 面向科学探究的动态认知角色协作框架, 通过角色先验、动态组队与 Exploration--Verification Alternation (EVA) 在假设生成与验证之间取得平衡. MBTI 在本项目中仅作为可解释的角色先验测试床, 并不代表对人类人格的主张.

## 特点
- 认知角色先验与角色划分
- 动态组队与意愿驱动的发言队列
- EVA 机制支持探索与验证交替

## 结构
- src/agents: 角色与代理管理, BaseAgent 统一逻辑, mbti/* 为 16 种角色实现, agentsmanager 负责类注册与加载
- src/workflows: DyCo 编排入口, discussion_workflow 组织四轮流程, mbtiagentsystem 负责系统装配与任务入口
- src/communication: message_manager 存储消息, speaking_queue 进行意愿驱动排队, team_manager 维护团队与代表
- src/prompts: 角色/功能/系统提示词模板, mbti 作为测试床, 预留 bigfive/enneagram/hexaco 扩展目录
- src/tools: knowledgemanager 负责知识库增删改查, system/websearch 提供检索与外部工具接口
- src/utils: llm_client 负责模型调用, discussion_summary_manager 汇总讨论, helpers 提供 XML 解析与格式化
- src/models: config/context/message/team 等数据结构, 约束配置与消息类型
- src/logger: logger_config 与公共日志工具, 统一任务级日志输出
- src/variables: tokens 统计与运行期变量记录

## 快速开始
1. 安装依赖: pip install -r requirements.txt
2. 在 src/workflows/mbtiagentsystem.py 中配置并运行
