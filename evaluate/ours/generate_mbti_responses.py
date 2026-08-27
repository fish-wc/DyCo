"""
批量运行 MBTI Agent System 处理 Deep Research Bench 数据集
读取 query.jsonl，对每个 query 生成回答并保存为 HTML 和元数据
"""
import sys,os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import json
import logging
import argparse
import time
from datetime import datetime
from typing import List, Dict, Optional

from src.workflows.mbtiagentsystem import MBTIAgentSystem
from src.variables.tokens import TOKEN_STATS, reset_token_stats, get_total_token_usage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MBTIResponseGenerator:
    """MBTI 系统批量响应生成器"""
    
    def __init__(
        self,
        query_file: str,
        output_dir: str,
        agent_ids: Optional[List[str]] = None,
        resume: bool = False
    ):
        """
        初始化生成器
        
        Args:
            query_file: query.jsonl 文件路径
            output_dir: 输出目录路径 (将保存 HTML 和 JSON 文件)
            agent_ids: 智能体 ID 列表 (默认使用论文中的 DyCo-Diverse 配置)
            resume: 是否恢复之前的运行 (跳过已完成的 query)
        """
        num_agents = 4
        self.query_file = Path(query_file)
        self.output_dir = Path(output_dir)
        # 这个部分是关于单个性格类型的
        if "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.no_mtbi":
            logger.info("实验模式: 不使用 MBTI 性格提示词")
            self.agent_ids = ['nomb_001']*len(agent_ids) if agent_ids else ['nomb_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.enfj":
            logger.info("实验模式: 全部使用 ENFJ 性格提示词")
            self.agent_ids = ['enfj_001']*len(agent_ids) if agent_ids else ['enfj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.enfp":
            logger.info("实验模式: 全部使用 ENFP 性格提示词")
            self.agent_ids = ['enfp_001']*len(agent_ids) if agent_ids else ['enfp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.entj":
            logger.info("实验模式: 全部使用 ENTJ 性格提示词")
            self.agent_ids = ['entj_001']*len(agent_ids) if agent_ids else ['entj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.entp":
            logger.info("实验模式: 全部使用 ENTP 性格提示词")
            self.agent_ids = ['entp_001']*len(agent_ids) if agent_ids else ['entp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.esfj":
            logger.info("实验模式: 全部使用 ESFJ 性格提示词")
            self.agent_ids = ['esfj_001']*len(agent_ids) if agent_ids else ['esfj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.esfp":
            logger.info("实验模式: 全部使用 ESFP 性格提示词")
            self.agent_ids = ['esfp_001']*len(agent_ids) if agent_ids else ['esfp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.estj":
            logger.info("实验模式: 全部使用 ESTJ 性格提示词")
            self.agent_ids = ['estj_001']*len(agent_ids) if agent_ids else ['estj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.estp":
            logger.info("实验模式: 全部使用 ESTP 性格提示词")
            self.agent_ids = ['estp_001']*len(agent_ids) if agent_ids else ['estp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.infj":
            logger.info("实验模式: 全部使用 INFJ 性格提示词")
            self.agent_ids = ['infj_001']*len(agent_ids) if agent_ids else ['infj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.infp":
            logger.info("实验模式: 全部使用 INFP 性格提示词")
            self.agent_ids = ['infp_001']*len(agent_ids) if agent_ids else ['infp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.intj":
            logger.info("实验模式: 全部使用 INTJ 性格提示词")
            self.agent_ids = ['intj_001']*len(agent_ids) if agent_ids else ['intj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.intp":
            logger.info("实验模式: 全部使用 INTP 性格提示词")
            self.agent_ids = ['intp_001']*len(agent_ids) if agent_ids else ['intp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.isfj":
            logger.info("实验模式: 全部使用 ISFJ 性格提示词")
            self.agent_ids = ['isfj_001']*len(agent_ids) if agent_ids else ['isfj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.isfp":
            logger.info("实验模式: 全部使用 ISFP 性格提示词")
            self.agent_ids = ['isfp_001']*len(agent_ids) if agent_ids else ['isfp_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.istj":
            logger.info("实验模式: 全部使用 ISTJ 性格提示词")
            self.agent_ids = ['istj_001']*len(agent_ids) if agent_ids else ['istj_001']*num_agents
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.istp":
            logger.info("实验模式: 全部使用 ISTP 性格提示词")
            self.agent_ids = ['istp_001']*len(agent_ids) if agent_ids else ['istp_001']*num_agents
        
        # 这个部分是关于混合性格类型的   
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.intj_intp_entj_entp":
            logger.info("实验模式: 使用 INTJ、INTP、ENTJ、ENTP 四种性格提示词")
            self.agent_ids = ['intj_001', 'intp_001', 'entj_001', 'entp_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.infj_infp_enfj_enfp":
            logger.info("实验模式: 使用 INFJ、INFP、ENFJ、ENFP 四种性格提示词")
            self.agent_ids = ['infj_001', 'infp_001', 'enfj_001', 'enfp_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.istj_estj_istp_estp":
            logger.info("实验模式: 使用 ISTJ、ESTJ、ISTP、ESTP 四种性格提示词")
            self.agent_ids = ['istj_001', 'estj_001', 'istp_001', 'estp_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.isfj_esfj_isfp_esfp":
            logger.info("实验模式: 使用 ISFJ、ESFJ、ISFP、ESFP 四种性格提示词")
            self.agent_ids = ['isfj_001', 'esfj_001', 'isfp_001', 'esfp_001']

        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.estj_istj_esfj_enfj":
            logger.info("实验模式: 使用 ESTJ、ISTJ、ESFJ、ENFJ 四种性格提示词")
            self.agent_ids = ['estj_001', 'istj_001', 'esfj_001', 'enfj_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.entj_intj_estp_isfj":
            logger.info("实验模式: 使用 ENTJ、INTJ、ESTP、ISFJ 四种性格提示词")
            self.agent_ids = ['entj_001', 'intj_001', 'estp_001', 'isfj_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "mbti.intp_entp_entj_istj":
            logger.info("实验模式: 使用 INTP、ENTP、ENTJ、ISTJ 四种性格提示词")
            self.agent_ids = ['intp_001', 'entp_001', 'entj_001', 'istj_001']

        # 消融实验
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "ablation.evam":
            logger.info("实验模式: Ablation - 去掉 EVAM 机制，默认使用entj_intj_estp_isfj四种性格，因为这四种性格是目前的sota的组合")
            self.agent_ids = ['entj_001', 'intj_001', 'estp_001', 'isfj_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "ablation.speaking_queue":
            logger.info("实验模式: Ablation - 去掉 Speaking Queue 机制，默认使用entj_intj_estp_isfj四种性格，因为这四种性格是目前的sota的组合")
            self.agent_ids = ['entj_001', 'intj_001', 'estp_001', 'isfj_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "ablation.round_1_all_discussion":
            logger.info("实验模式: Ablation - 去掉 Round 1 All Discussion 机制，默认使用entj_intj_estp_isfj四种性格，因为这四种性格是目前的sota的组合")
            self.agent_ids = ['entj_001', 'intj_001', 'estp_001', 'isfj_001']
        elif "exp_mode" in os.environ and os.environ["exp_mode"] == "ablation.round_3_team_discussion":
            logger.info("实验模式: Ablation - 去掉 Round 3 Team Discussion 机制，默认使用entj_intj_estp_isfj四种性格，因为这四种性格是目前的sota的组合")
            self.agent_ids = ['entj_001', 'intj_001', 'estp_001', 'isfj_001']
    
    
        # 默认使用论文中的 DyCo-Diverse 配置：ENTJ、INTJ、ESTP、ISFJ
        else:
            self.agent_ids = agent_ids or ["entj_001", "intj_001", "estp_001", "isfj_001"]
        self.resume = resume
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载所有 queries
        self.queries = self._load_queries()
        logger.info(f"加载了 {len(self.queries)} 个 queries")
        
        # 如果是恢复模式，加载已完成的 query_id
        self.completed_ids = set()
        if self.resume:
            self.completed_ids = self._load_completed_ids()
            logger.info(f"恢复模式：已完成 {len(self.completed_ids)} 个 queries")
    
    def _load_queries(self) -> List[Dict]:
        """加载 query.jsonl 文件"""
        queries = []
        
        if not self.query_file.exists():
            raise FileNotFoundError(f"Query 文件不存在: {self.query_file}")
        
        with open(self.query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        query = json.loads(line)
                        queries.append(query)
                    except json.JSONDecodeError as e:
                        logger.error(f"解析 JSON 失败: {e}")
                        continue
        
        return queries
    
    def _load_completed_ids(self) -> set:
        """加载已完成的 query_id (通过检查输出文件)"""
        completed = set()
        
        # 检查 HTML 和 JSON 文件是否都存在
        for query in self.queries:
            query_id = query.get('id')
            if query_id is None:
                continue
            
            html_file = self.output_dir / f"{query_id}.html"
            json_file = self.output_dir / f"{query_id}.json"
            
            if html_file.exists() and json_file.exists():
                completed.add(query_id)
        
        return completed
    
    def process_query(self, query: Dict) -> Dict:
        """
        处理单个 query
        
        Args:
            query: query 字典，包含 id, prompt 等字段
            
        Returns:
            处理结果字典
        """
        query_id = query.get('id')
        prompt = query.get('prompt')
        # topic = query.get('topic', 'common')
        # 使用通用提示词。
        topic = "Common"
        
        if query_id is None or not prompt:
            raise ValueError(f"Query 缺少必要字段: {query}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"开始处理 Query {query_id}")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"{'='*80}\n")
        
        # 重置 token 统计
        reset_token_stats()
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用 query_id 作为 task_id
        task_id = f"{query_id}"
        
        try:
            # 初始化 MBTI Agent System
            logger.info(f"初始化 MBTI Agent System (task_id={task_id})...")
            system = MBTIAgentSystem(
                task_id=task_id,
                agent_ids=self.agent_ids
            )
            
            # 运行任务
            logger.info("开始运行工作流...")
            workflow_result = system.solve_task(
                task=prompt,
                topic=topic,
                initial_agent_id=self.agent_ids[0]
            )
            
            # 记录结束时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 获取 token 使用情况
            token_usage = get_total_token_usage()
            
            # 提取结果
            success = workflow_result.get('success', False)
            final_answer = workflow_result.get('final_answer', '')
            
            # 提取 HTML 格式
            article_html = ''
            article_text = ''
            
            if isinstance(final_answer, dict):
                article_html = final_answer.get('article_html', '')
                article_text = final_answer.get('article_text', '')
            elif isinstance(final_answer, str):
                article_text = final_answer
                article_html = f"<article>{final_answer}</article>"
            
            # 构建元数据
            metadata = {
                'query_id': query_id,
                'task_id': task_id,
                'prompt': prompt,
                'topic': query.get('topic', ''),
                'language': query.get('language', 'zh'),
                'success': success,
                'elapsed_time_seconds': elapsed_time,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'token_usage': token_usage,
                'token_history': TOKEN_STATS.get('history', []),
                'workflow_path': workflow_result.get('workflow_path', ''),
                'total_rounds': workflow_result.get('total_rounds', 0),
                'final_representative': workflow_result.get('final_representative', ''),
                'article_length': len(article_text),
                'agent_ids': self.agent_ids,
            }
            
            

            # 如果失败，记录错误信息
            if not success:
                metadata['error_message'] = workflow_result.get('error_message', 'Unknown error')
            
            # 保存 HTML 文件
            html_file = self.output_dir / f"{query_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(article_html)
            logger.info(f"✓ HTML 已保存: {html_file}")
            
            # 保存元数据 JSON 文件
            json_file = self.output_dir / f"{query_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 元数据已保存: {json_file}")
            
            # 保存纯文本文件
            txt_file = self.output_dir / f"{query_id}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(article_text)

            # 打印摘要
            logger.info(f"\n{'='*80}")
            logger.info(f"Query {query_id} 处理完成")
            logger.info(f"状态: {'✅ 成功' if success else '❌ 失败'}")
            logger.info(f"耗时: {elapsed_time:.2f} 秒")
            logger.info(f"Token 使用: {token_usage['total']} (prompt: {token_usage['prompt']}, completion: {token_usage['completion']})")
            logger.info(f"文章长度: {len(article_text)} 字符")
            logger.info(f"{'='*80}\n")
            
            return metadata
            
        except Exception as e:
            # 记录错误
            end_time = time.time()
            elapsed_time = end_time - start_time
            token_usage = get_total_token_usage()
            
            error_metadata = {
                'query_id': query_id,
                'task_id': task_id,
                'prompt': prompt,
                'topic': query.get('topic', ''),
                'language': query.get('language', 'zh'),
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'elapsed_time_seconds': elapsed_time,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'token_usage': token_usage,
                'agent_ids': self.agent_ids,
            }
            
            # 保存错误信息
            json_file = self.output_dir / f"{query_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(error_metadata, f, ensure_ascii=False, indent=2)
            
            logger.error(f"❌ Query {query_id} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return error_metadata
    
    def run(self, query_ids: Optional[List[int]] = None, limit: Optional[int] = None):
        """
        批量处理 queries
        
        Args:
            query_ids: 指定要处理的 query_id 列表 (None 表示处理所有)
            limit: 限制处理数量 (None 表示无限制)
        """
        # 筛选要处理的 queries
        queries_to_process = self.queries
        
        if query_ids is not None:
            query_ids_set = set(query_ids)
            queries_to_process = [q for q in queries_to_process if q.get('id') in query_ids_set]
            logger.info(f"指定处理 {len(query_ids)} 个 queries: {query_ids}")
        
        # 如果是恢复模式，过滤已完成的
        if self.resume:
            queries_to_process = [q for q in queries_to_process if q.get('id') not in self.completed_ids]
            logger.info(f"恢复模式：剩余 {len(queries_to_process)} 个 queries 待处理")
        
        # 应用 limit
        if limit is not None and limit > 0:
            queries_to_process = queries_to_process[:limit]
            logger.info(f"限制处理数量: {limit}")
        
        if not queries_to_process:
            logger.info("没有需要处理的 queries")
            return
        
        logger.info(f"\n开始批量处理 {len(queries_to_process)} 个 queries\n")
        
        # 统计信息
        total = len(queries_to_process)
        success_count = 0
        failed_count = 0
        results = []
        
        # 逐个处理
        for i, query in enumerate(queries_to_process, 1):
            logger.info(f"\n进度: {i}/{total}")
            
            try:
                result = self.process_query(query)
                results.append(result)
                
                if result.get('success', False):
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"处理 query {query.get('id')} 时发生意外错误: {e}")
                failed_count += 1
                continue
        
        # 打印最终统计
        logger.info(f"\n{'='*80}")
        logger.info("批量处理完成")
        logger.info(f"总计: {total}")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {failed_count}")
        logger.info(f"输出目录: {self.output_dir.absolute()}")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='批量运行 MBTI Agent System 处理 Deep Research Bench 数据集'
    )
    
    # 输入输出参数
    parser.add_argument(
        '--query_file',
        type=str,
        default='evaluate/deepresearch/deep_research_bench-main/data/prompt_data/query.jsonl',
        help='Query 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluate/deepresearch/deep_research_bench-main/data/test_data/raw_data/MBTI_MAS',
        help='输出目录路径'
    )
    
    # 选择性处理参数
    parser.add_argument(
        '--query_ids',
        type=int,
        nargs='+',
        default=None,
        help='指定要处理的 query_id (例如: --query_ids 1 15 16)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理数量 (用于测试)'
    )
    
    # 恢复模式
    parser.add_argument(
        '--resume',
        action='store_true',
        help='恢复之前的运行，跳过已完成的 queries'
    )
    
    # 智能体配置
    parser.add_argument(
        '--agent_ids',
        type=str,
        nargs='+',
        default=["entj_001", "intj_001", "estp_001", "isfj_001"],
        help='智能体 ID 列表'
    )
    
    # 设置实验模式exp_mode
    parser.add_argument(
        '--exp_mode',
        type=str,
        default='',
        help='''实验模式 (可选: mbti.no_mtbi，表示不使用 MBTI 性格。
        mbti.enfj 表示全部使用 ENFJ 性格，mbti.enfp 表示全部使用 ENFP 性格
        mbti.entj表示全部使用 ENTJ 性格，mbti.entp 表示全部使用 ENTP 性格)
        mbti.esfj表示全部使用 ESFJ 性格，mbti.esfp 表示全部使用 ESFP 性格,
        mbti.estj表示全部使用 ESTJ 性格，mbti.estp 表示全部使用 ESTP 性格,
        mbti.infj表示全部使用 INFJ 性格，mbti.infp 表示全部使用 INFP 性格,
        mbti.intj表示全部使用 INTJ 性格，mbti.intp 表示全部使用 INTP 性格,
        mbti.isfj表示全部使用 ISFJ 性格，mbti.isfp 表示全部使用 ISFP 性格,
        mbti.istj表示全部使用 ISTJ 性格，mbti.istp 表示全部使用 ISTP 性格)
        
        mbti.intj_intp_entj_entp 表示 ENFJ、ENFP、ENTJ、ENTP 四种性格各一个，
        mbti.infj_infp_enfj_enfp 表示 INFJ、INFP、ENFJ、ENFP 四种性格各一个,
        mbti.istj_estj_istp_estp 表示 ISTJ、ESTJ、ISTP、ESTP 四种性格各一个
        mbti.isfj_esfj_isfp_esfp 表示 ISFJ、ESFJ、ISFP、ESFP 四种性格各一个,
        
        mbti.entj_intj_estp_isfj 表示 ENTJ、INTJ、ESTP、ISFJ 四种性格各一个,
        mbti.intp_entp_entj_istj 表示 INTP、ENTP、ENTJ、ISTJ 四种性格各一个,
        mbti.estj_istj_esfj_enfj 表示 ESTJ、ISTJ、ESFJ、ENFJ 四种性格各一个,
        ''',
    )
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.exp_mode:
        logger.info(f"设置实验模式: {args.exp_mode}")
        import os
        os.environ['exp_mode'] = args.exp_mode
    
    # 转换相对路径为绝对路径
    project_root = Path(__file__).resolve().parents[2]
    query_file = project_root / args.query_file
    output_dir = project_root / args.output_dir
    
    # 创建生成器
    generator = MBTIResponseGenerator(
        query_file=str(query_file),
        output_dir=str(output_dir),
        agent_ids=args.agent_ids,
        resume=args.resume
    )
    
    # 运行
    generator.run(
        query_ids=args.query_ids,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
