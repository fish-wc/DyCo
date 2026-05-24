"""
工具函数
"""
from pathlib import Path
import re
import json
import ast
from typing import Dict, Any, List
from .json_parser import extract_json_from_text

def ensure_directory_exists(path: str):
    """
    确保目录存在,不存在则创建
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_task_id() -> str:
    """
    生成任务ID
    
    Returns:
        任务ID
    """
    import uuid
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"task_{timestamp}_{unique_id}"


def get_tag_content(content, tag: str = "ANSWER", is_json: bool = True) -> dict:
    """
    提取内容中的特定标签内容,同时返回去除标签后的原始文本。
    
    Args:
        content: 原始内容
        tag: 需要提取的标签名称(字符串或列表)
        is_json: 是否将提取内容解析为JSON格式
    Returns:
        包含解析结果及去除标签内容的字典
        - 如果tag是列表: {"tag1": "内容1", "tag2": "内容2", "remaining_content": "剩余内容"}
        - 如果tag是字符串: {"result": "标签内容", "content": "去掉标签后的内容"}
    """
    # 如果tag是列表,提取所有标签并返回
    if isinstance(tag, list):
        result = {}
        remaining_content = content
        for t in tag:
            extracted = get_tag_content(remaining_content, t, is_json)
            # 提取的标签内容存储到对应的key中
            result[t.lower()] = extracted.get("result", "")
            # 更新剩余内容(去掉已提取的标签)
            remaining_content = extracted.get("content", remaining_content)
        # 最终剩余的内容存储为remaining_content
        result["remaining_content"] = remaining_content
        return result
    
    answer = {}
    pattern = fr"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, content, re.DOTALL)
    cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL)

    if not matches:
        answer["content"] = cleaned_content
        return answer

    if not is_json:
        # 可能会有多个匹配项，取出来拼接即可
        combined_content = "\n".join(match.strip() for match in matches)
        answer["result"] = combined_content
        answer["content"] = cleaned_content
        return answer
    
    ############### 处理json ###############
    for raw_match in matches:
        candidate = raw_match.strip()
        if not candidate:
            continue

        # 先尝试直接解析 JSON
        try:
            parsed_match = json.loads(candidate)
            answer.update(parsed_match if isinstance(parsed_match, dict) else {"value": parsed_match})
            continue
        except Exception:
            pass

        # 移除可能的围绕反引号或多余的修饰
        candidate_clean = candidate.strip("`\n ")

        # 尝试使用 Python 字面量解析（兼容单引号）
        try:
            literal_value = ast.literal_eval(candidate_clean)
            if isinstance(literal_value, dict):
                answer.update(literal_value)
                continue
        except Exception:
            pass

        # 最后尝试从文本中提取 JSON 片段
        extracted = extract_json_from_text(candidate_clean, strict=False)
        if extracted:
            answer.update(extracted)
            continue

        print("Error parsing JSON from tag content: 无法解析匹配的标签内容")

    answer["content"] = cleaned_content
    return answer


def merge_citations(sections_data: list) -> tuple:
    """
    统一处理多个章节的参考文献编号,去重并重新编号
    
    Args:
        sections_data: 章节数据列表,每项包含 {'content': str, 'citations': str}
    
    Returns:
        (处理后的章节内容列表, 统一的参考文献列表字符串)
    """
    import re
    
    # 存储全局引用映射
    global_citations = []  # [(url, description)]
    url_to_index = {}  # url -> 全局编号
    
    merged_content = []
    
    for section in sections_data:
        section_content = section.get('content', '')
        section_citations = section.get('citations', '')
        
        if not section_content:
            continue
        
        # 解析该章节的引用列表
        # 格式: [1] 描述 | URL
        citation_pattern = r'\[(\d+)\]\s*(.+?)\s*\|\s*(.+?)(?=\n\[|\n*$)'
        local_citations = re.findall(citation_pattern, section_citations, re.DOTALL)
        
        # 建立局部编号到全局编号的映射
        local_to_global = {}
        
        for local_idx, desc, url in local_citations:
            url = url.strip()
            desc = desc.strip()
            
            # 检查URL是否已存在
            if url in url_to_index:
                # 已存在,使用现有编号
                local_to_global[int(local_idx)] = url_to_index[url]
            else:
                # 新URL,分配新编号
                global_idx = len(global_citations) + 1
                global_citations.append((url, desc))
                url_to_index[url] = global_idx
                local_to_global[int(local_idx)] = global_idx
        
        # 替换正文中的引用编号
        updated_content = section_content
        
        # 按编号从大到小替换,避免替换冲突(如[1]和[11])
        for local_idx in sorted(local_to_global.keys(), reverse=True):
            global_idx = local_to_global[local_idx]
            # 使用负向预查和负向回顾,确保不会匹配到更长的数字
            pattern = r'(?<!\d)\[' + str(local_idx) + r'\](?!\d)'
            updated_content = re.sub(pattern, f'[{global_idx}]', updated_content)
        
        merged_content.append(updated_content)
    
    # 生成最终的参考文献列表
    final_citations = []
    for idx, (url, desc) in enumerate(global_citations, 1):
        final_citations.append(f'[{idx}] {desc} | {url}')
    
    return merged_content, '\n'.join(final_citations)


def sanitize_text_for_json(s: str) -> str:
    """
    Make a string safe for JSON serialization by normalizing newlines,
    escaping lone backslashes, and removing problematic control characters.

    This is defensive: data coming from LLMs or web scrapers can contain
    stray backslashes or control bytes that later break downstream JSON
    parsers which expect valid JSON escapes.
    """
    if s is None:
        return s
    # Ensure it's a str
    if not isinstance(s, str):
        s = str(s)

    # Normalize CRLF to LF
    s = s.replace('\r\n', '\n').replace('\r', '\n')

    # Escape lone backslashes (\ -> \\)
    s = s.replace('\\', '\\\\')

    # Remove control characters except common whitespace (tab, newline)
    cleaned_chars = []
    for ch in s:
        code = ord(ch)
        if code >= 0x20 or ch in ('\n', '\t'):
            cleaned_chars.append(ch)
        else:
            # replace with space to keep offsets stable
            cleaned_chars.append(' ')
    return ''.join(cleaned_chars)


def extract_xml_tag_content(text: str, tag: str) -> str:
    """
    从文本中提取单个 XML 标签的内容（支持带属性的标签）
    
    Args:
        text: 原始文本
        tag: 标签名称（不含尖括号）
    
    Returns:
        标签内容（去除首尾空白）
    
    示例:
        extract_xml_tag_content('<TITLE>标题</TITLE>', 'TITLE') -> '标题'
        extract_xml_tag_content('<CHUNK index="1" similarity="0.8">内容</CHUNK>', 'CHUNK') -> '内容'
    """
    # 支持带属性的标签，如 <CHUNK index="1"> 或 <RESULT index="1">
    pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_all_xml_blocks(text: str, block_tag: str) -> list:
    """
    从文本中提取所有指定的 XML 块（支持带属性的标签）
    
    Args:
        text: 原始文本
        block_tag: 块标签名称（如 'KNOWLEDGE_POINT', 'RESULT', 'CHUNK'）
    
    Returns:
        完整的XML块列表（包含标签和内容）
    """
    # 支持带属性的标签，如 <RESULT index="1"> 或 <CHUNK similarity="0.5">
    pattern = rf"<{block_tag}[^>]*>.*?</{block_tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def parse_knowledge_points(text: str) -> list:
    """
    从 LLM 输出中解析所有知识点
    
    Args:
        text: LLM 输出的文本（包含 <ANALYSIS_RESULT> 和 <KNOWLEDGE_POINT> 标签）
    
    Returns:
        知识点列表，每项为字典 {'title': str, 'content': str, 'importance': str}
    """
    knowledge_points = []
    
    # 首先提取 ANALYSIS_RESULT 内容
    analysis_content = extract_xml_tag_content(text, "ANALYSIS_RESULT")
    if not analysis_content:
        # 如果没有找到 ANALYSIS_RESULT，尝试直接在原文中查找
        analysis_content = text
    
    # 提取所有 KNOWLEDGE_POINT 块
    kp_blocks = extract_all_xml_blocks(analysis_content, "KNOWLEDGE_POINT")
    
    for block in kp_blocks:
        title = extract_xml_tag_content(block, "TITLE")
        content = extract_xml_tag_content(block, "CONTENT")
        importance = extract_xml_tag_content(block, "IMPORTANCE")
        
        if title and content:  # 至少要有标题和内容
            knowledge_points.append({
                'title': title,
                'content': content,
                'importance': importance if importance else 'medium'
            })
    
    return knowledge_points


def build_knowledge_with_meta(knowledge_content: str, agent_id: str, mbti_type: str, 
                              task_id: str = None, knowledge_type: str = "analysis",
                              importance: str = "medium", additional_meta: dict = None) -> dict:
    """
    构建带有 meta 信息的知识字典（用于向量数据库存储）
    
    改进策略：分离存储
    - content: 纯净的知识内容（用于 embedding）
    - metadata: 所有 meta 信息（作为元数据存储）
    
    这样可以避免 XML 标签稀释语义密度，提高检索准确度。
    
    Args:
        knowledge_content: 核心知识内容（纯文本，不含XML标签）
        agent_id: 智能体ID
        mbti_type: MBTI 类型
        task_id: 任务ID（可选）
        knowledge_type: 知识类型（如 'analysis', 'discussion', 'conclusion'）
        importance: 重要性（high/medium/low）
        additional_meta: 额外的元数据字典
    
    Returns:
        包含 content 和 metadata 的字典
        {
            'content': '纯净的知识内容（用于embedding）',
            'metadata': {'agent_id': ..., 'mbti_type': ..., ...}
        }
    """
    from datetime import datetime
    
    # 构建 metadata 字典
    metadata = {
        'agent_id': agent_id,
        'mbti_type': mbti_type,
        'knowledge_type': knowledge_type,
        'importance': importance,
        'timestamp': datetime.now().isoformat()
    }
    
    if task_id:
        metadata['task_id'] = task_id
    
    if additional_meta:
        metadata.update(additional_meta)
    
    return {
        'content': knowledge_content,  # 纯净内容，只用于 embedding
        'metadata': metadata  # 所有 meta 信息
    }


def parse_evaluation_result(text: str) -> dict:
    """
    从 LLM 输出中解析评审结果
    
    Args:
        text: LLM 输出的文本（包含 <EVALUATION_RESULT> 标签）
    
    Returns:
        评审结果字典，包含：
        {
            'decision': {
                'approval_status': str,  # 通过/有条件通过/建议修改/拒绝
                'confidence_level': int,  # 0-100
                'summary': str
            },
            'evaluation_points': [
                {
                    'title': str,
                    'content': str,
                    'dimension': str,
                    'importance': str
                },
                ...
            ]
        }
    """
    result = {
        'decision': {},
        'evaluation_points': []
    }
    
    # 提取 EVALUATION_RESULT 内容
    eval_content = extract_xml_tag_content(text, "EVALUATION_RESULT")
    if not eval_content:
        eval_content = text
    
    # 解析决策部分
    decision_block = extract_xml_tag_content(eval_content, "DECISION")
    if decision_block:
        result['decision'] = {
            'approval_status': extract_xml_tag_content(decision_block, "APPROVAL_STATUS"),
            'confidence_level': extract_xml_tag_content(decision_block, "CONFIDENCE_LEVEL"),
            'summary': extract_xml_tag_content(decision_block, "ONE_SENTENCE_SUMMARY")
        }
        
        # 转换 confidence_level 为整数
        try:
            result['decision']['confidence_level'] = int(result['decision']['confidence_level'])
        except (ValueError, TypeError):
            result['decision']['confidence_level'] = 0
    
    # 提取所有 EVALUATION_POINT 块
    ep_blocks = extract_all_xml_blocks(eval_content, "EVALUATION_POINT")
    
    for block in ep_blocks:
        title = extract_xml_tag_content(block, "TITLE")
        content = extract_xml_tag_content(block, "CONTENT")
        dimension = extract_xml_tag_content(block, "DIMENSION")
        importance = extract_xml_tag_content(block, "IMPORTANCE")
        
        if title and content:
            result['evaluation_points'].append({
                'title': title,
                'content': content,
                'dimension': dimension if dimension else 'general',
                'importance': importance if importance else 'medium'
            })
    
    return result


def parse_team_preference_result(text: str) -> dict:
    """
    从 LLM 输出中解析组队偏好结果（支持多种 MBTI 类型的不同评估维度）
    
    支持的 MBTI 评估维度:
    - ENFJ: COLLABORATION_POTENTIAL, CAPABILITY_COMPLEMENT, COMMUNICATION_FIT, GROWTH_VALUE (30%, 30%, 20%, 20%)
    - ENFP: INNOVATION_POTENTIAL, TASK_ALIGNMENT, ANALYTICAL_PERSPECTIVE, COLLABORATION_FIT (35%, 25%, 25%, 15%)
    - ENTJ: STRATEGIC_CONTRIBUTION, PROFESSIONAL_CAPABILITY, EXECUTION_POWER, SYSTEM_COMPATIBILITY (40%, 30%, 20%, 10%)
    - ENTP: THINKING_DEPTH, INNOVATION, LOGICAL_RIGOR, DEBATE_VALUE (35%, 30%, 25%, 10%)
    
    Args:
        text: LLM 输出的文本
    
    Returns:
        解析结果字典：
        {
            'overall_strategy': str,
            'candidates': [{'agent_id': str, 'scores': {...}, 'preference_level': str, ...}],
            'recommendations': str,
            'suggestions': [str]
        }
    """
    # 程度词到分数的映射
    DEGREE_MAPPING = {
        '极高': 10, '很高': 9, '较高': 8, '高': 7.5,
        '中等偏上': 7, '中等': 6, '中等偏下': 5,
        '较低': 4, '低': 3, '很低': 2, '极低': 1
    }
    
    PREFERENCE_MAPPING = {
        '强烈推荐': 9.0, '推荐': 7.5, '可以考虑': 6.0,
        '谨慎考虑': 4.5, '不推荐': 2.0
    }
    
    # 不同MBTI类型的评估维度配置 (标签名, 权重, 输出字段名)
    DIMENSION_CONFIGS = {
        # ENFJ 维度
        'COLLABORATION_POTENTIAL': ('collaboration_potential', 0.3),
        'CAPABILITY_COMPLEMENT': ('capability_complement', 0.3),
        'COMMUNICATION_FIT': ('communication_fit', 0.2),
        'GROWTH_VALUE': ('growth_value', 0.2),
        
        # ENFP 维度
        'INNOVATION_POTENTIAL': ('innovation_potential', 0.35),
        'TASK_ALIGNMENT': ('task_alignment', 0.25),
        'ANALYTICAL_PERSPECTIVE': ('analytical_perspective', 0.25),
        'COLLABORATION_FIT': ('collaboration_fit', 0.15),
        
        # ENTJ 维度
        'STRATEGIC_CONTRIBUTION': ('strategic_contribution', 0.4),
        'PROFESSIONAL_CAPABILITY': ('professional_capability', 0.3),
        'EXECUTION_POWER': ('execution_power', 0.2),
        'SYSTEM_COMPATIBILITY': ('system_compatibility', 0.1),
        
        # ENTP 维度
        'THINKING_DEPTH': ('thinking_depth', 0.35),
        'INNOVATION': ('innovation', 0.3),
        'LOGICAL_RIGOR': ('logical_rigor', 0.25),
        'DEBATE_VALUE': ('debate_value', 0.1),
    }
    
    result = {
        'overall_strategy': '',
        'candidates': [],
        'recommendations': '',
        'suggestions': []
    }
    
    # 提取主内容
    main_content = extract_xml_tag_content(text, "TEAM_PREFERENCE_RESULT")
    if not main_content:
        main_content = text
    
    # 提取总体策略
    result['overall_strategy'] = extract_xml_tag_content(main_content, "OVERALL_STRATEGY") or ''
    
    # 提取候选人评估
    evals_content = extract_xml_tag_content(main_content, "CANDIDATE_EVALUATIONS")
    if evals_content:
        candidate_blocks = extract_all_xml_blocks(evals_content, "CANDIDATE")
        
        for block in candidate_blocks:
            agent_id = extract_xml_tag_content(block, "AGENT_ID")
            if not agent_id:
                continue
            
            comment = extract_xml_tag_content(block, "BRIEF_COMMENT") or ''
            preference = extract_xml_tag_content(block, "PREFERENCE_LEVEL") or '可以考虑'
            preference_score = PREFERENCE_MAPPING.get(preference.strip(), 6.0)
            
            # 动态提取评估维度
            scores = {}
            total_score = 0
            found_dimensions = []
            
            for tag_name, (field_name, weight) in DIMENSION_CONFIGS.items():
                degree_word = extract_xml_tag_content(block, tag_name)
                if degree_word:
                    score = DEGREE_MAPPING.get(degree_word.strip(), 6)
                    scores[field_name] = score
                    total_score += score * weight
                    found_dimensions.append((field_name, score, weight))
            
            # 如果没有找到任何维度，使用默认值
            if not found_dimensions:
                # 尝试检测是哪种类型并使用默认维度
                scores = {
                    'dimension_1': 6,
                    'dimension_2': 6,
                    'dimension_3': 6,
                    'dimension_4': 6
                }
                total_score = 6.0
            
            scores['total_score'] = round(total_score, 2)
            
            result['candidates'].append({
                'agent_id': agent_id.strip(),
                'scores': scores,
                'preference_level': preference.strip(),
                'preference_score': preference_score,
                'brief_comment': comment.strip()
            })
    
    # 提取推荐
    result['recommendations'] = extract_xml_tag_content(main_content, "RECOMMENDATIONS") or ''
    
    # 提取额外建议
    suggestions_content = extract_xml_tag_content(main_content, "ADDITIONAL_SUGGESTIONS")
    if suggestions_content:
        suggestion_blocks = extract_all_xml_blocks(suggestions_content, "SUGGESTION")
        for block in suggestion_blocks:
            suggestion_text = extract_xml_tag_content(block, "SUGGESTION")
            if suggestion_text and suggestion_text.strip():
                result['suggestions'].append(suggestion_text.strip())
    
    return result


def parse_attitude_result(text: str) -> dict:
    """
    从 LLM 输出中解析 attitude 结果
    
    Args:
        text: LLM 输出的文本（包含 <ATTITUDE_RESULT> 标签）
    
    Returns:
        态度分析结果字典，包含：
        {
            'discussion_evaluation': {
                'completeness_degree': str,  # 完善度程度词
                'completeness_analysis': str,  # 完善度分析
                'key_issues': str,  # 关键问题
                'consensus_status': str  # 共识状态
            },
            'termination_suggestion': {
                'should_terminate': bool,  # 是否建议结束
                'termination_reason': str,  # 理由
                'confidence_level': str  # 确定程度
            },
            'speaking_intention': {
                'desire_to_speak': bool,  # 是否想发言
                'intention_degree': str,  # 意愿程度词
                'intention_score': float,  # 意愿分数（0-10）
                'intention_reason': str  # 理由
            },
            'representative_intention': {
                'desire_to_represent': bool,  # 是否愿意担任代表
                'representative_degree': str,  # 代表意愿程度词
                'representative_score': float,  # 代表意愿分数（0-10）
                'representative_reason': str  # 理由
            },
            'speaking_plan': {
                'core_viewpoint': str,  # 核心观点
                'key_points': str,  # 关键论点
                'emphasis_aspects': str  # 强调方面
            },
            'knowledge_content': str  # 用于存入知识库的内容（不含发言思路）
        }
    """
    # 发言意愿程度词映射（0-10分）
    INTENTION_DEGREE_MAPPING = {
        '极强': 10.0,
        '很强': 9.0,
        '较强': 8.0,
        '中等偏强': 7.0,
        '中等': 6.0,
        '中等偏弱': 5.0,
        '较弱': 4.0,
        '很弱': 3.0,
        '极弱': 2.0
    }
    
    # 代表意愿程度词映射（0-10分）
    REPRESENTATIVE_DEGREE_MAPPING = {
        '极强': 10.0,
        '很强': 9.0,
        '较强': 8.0,
        '中等': 6.0,
        '较弱': 4.0,
        '很弱': 3.0,
        '极弱': 2.0
    }
    
    result = {
        'discussion_evaluation': {},
        'termination_suggestion': {},
        'speaking_intention': {},
        'representative_intention': {},
        'speaking_plan': {},
        'knowledge_content': ''
    }
    
    # 提取 ATTITUDE_RESULT 内容
    attitude_content = extract_xml_tag_content(text, "ATTITUDE_RESULT")
    if not attitude_content:
        return result
    
    # 1. 解析讨论评估部分
    discussion_block = extract_xml_tag_content(attitude_content, "DISCUSSION_EVALUATION")
    if discussion_block:
        result['discussion_evaluation'] = {
            'completeness_degree': extract_xml_tag_content(discussion_block, "COMPLETENESS_DEGREE").strip(),
            'completeness_analysis': extract_xml_tag_content(discussion_block, "COMPLETENESS_ANALYSIS").strip(),
            'key_issues': extract_xml_tag_content(discussion_block, "KEY_ISSUES").strip(),
            'consensus_status': extract_xml_tag_content(discussion_block, "CONSENSUS_STATUS").strip()
        }
    
    # 2. 解析结束建议部分
    termination_block = extract_xml_tag_content(attitude_content, "TERMINATION_SUGGESTION")
    if termination_block:
        should_terminate_text = extract_xml_tag_content(termination_block, "SHOULD_TERMINATE").strip()
        result['termination_suggestion'] = {
            'should_terminate': should_terminate_text == '是',
            'termination_reason': extract_xml_tag_content(termination_block, "TERMINATION_REASON").strip(),
            'confidence_level': extract_xml_tag_content(termination_block, "CONFIDENCE_LEVEL").strip()
        }
    
    # 3. 解析发言意愿部分
    intention_block = extract_xml_tag_content(attitude_content, "SPEAKING_INTENTION")
    if intention_block:
        desire_text = extract_xml_tag_content(intention_block, "DESIRE_TO_SPEAK").strip()
        intention_degree = extract_xml_tag_content(intention_block, "INTENTION_DEGREE").strip()
        
        # 将程度词映射为分数
        intention_score = INTENTION_DEGREE_MAPPING.get(intention_degree, 6.0)
        
        result['speaking_intention'] = {
            'desire_to_speak': desire_text == '是',
            'intention_degree': intention_degree,
            'intention_score': intention_score,
            'intention_reason': extract_xml_tag_content(intention_block, "INTENTION_REASON").strip()
        }
    
    # 4. 解析代表发言意愿部分（新增）
    representative_block = extract_xml_tag_content(attitude_content, "REPRESENTATIVE_INTENTION")
    if representative_block:
        desire_repr_text = extract_xml_tag_content(representative_block, "DESIRE_TO_REPRESENT").strip()
        representative_degree = extract_xml_tag_content(representative_block, "REPRESENTATIVE_DEGREE").strip()
        
        # 将程度词映射为分数
        representative_score = REPRESENTATIVE_DEGREE_MAPPING.get(representative_degree, 6.0)
        
        result['representative_intention'] = {
            'desire_to_represent': desire_repr_text == '是',
            'representative_degree': representative_degree,
            'representative_score': representative_score,
            'representative_reason': extract_xml_tag_content(representative_block, "REPRESENTATIVE_REASON").strip()
        }
    
    # 5. 解析发言计划部分
    plan_block = extract_xml_tag_content(attitude_content, "SPEAKING_PLAN")
    if plan_block:
        result['speaking_plan'] = {
            'core_viewpoint': extract_xml_tag_content(plan_block, "CORE_VIEWPOINT").strip(),
            'key_points': extract_xml_tag_content(plan_block, "KEY_POINTS").strip(),
            'emphasis_aspects': extract_xml_tag_content(plan_block, "EMPHASIS_ASPECTS").strip()
        }
    
    # 6. 构建知识库内容（不含发言计划）
    knowledge_parts = []
    
    # 讨论评估
    if result['discussion_evaluation']:
        eval_text = f"【讨论完善度评估】\n"
        eval_text += f"完善程度: {result['discussion_evaluation']['completeness_degree']}\n"
        eval_text += f"分析: {result['discussion_evaluation']['completeness_analysis']}\n"
        eval_text += f"关键问题: {result['discussion_evaluation']['key_issues']}\n"
        eval_text += f"共识状态: {result['discussion_evaluation']['consensus_status']}"
        knowledge_parts.append(eval_text)
    
    # 结束建议
    if result['termination_suggestion']:
        term_text = f"【结束讨论建议】\n"
        term_text += f"建议结束: {'是' if result['termination_suggestion']['should_terminate'] else '否'}\n"
        term_text += f"理由: {result['termination_suggestion']['termination_reason']}\n"
        term_text += f"确定程度: {result['termination_suggestion']['confidence_level']}"
        knowledge_parts.append(term_text)
    
    # 发言意愿（不含具体计划）
    if result['speaking_intention']:
        intent_text = f"【发言意愿】\n"
        intent_text += f"是否想发言: {'是' if result['speaking_intention']['desire_to_speak'] else '否'}\n"
        intent_text += f"意愿程度: {result['speaking_intention']['intention_degree']} (分数: {result['speaking_intention']['intention_score']})\n"
        intent_text += f"理由: {result['speaking_intention']['intention_reason']}"
        knowledge_parts.append(intent_text)
    
    # 代表发言意愿（新增）
    if result['representative_intention']:
        repr_text = f"【代表发言意愿】\n"
        repr_text += f"是否愿意担任代表: {'是' if result['representative_intention']['desire_to_represent'] else '否'}\n"
        repr_text += f"意愿程度: {result['representative_intention']['representative_degree']} (分数: {result['representative_intention']['representative_score']})\n"
        repr_text += f"理由: {result['representative_intention']['representative_reason']}"
        knowledge_parts.append(repr_text)
    
    result['knowledge_content'] = '\n\n'.join(knowledge_parts)
    
    return result


def parse_generation_result(text: str) -> Dict[str, Any]:
    """
    解析 generate 方法的 XML 输出结果
    
    预期 XML 结构:
    <GENERATION_RESULT>
      <THINKING>
        <CONTEXT_UNDERSTANDING>...</CONTEXT_UNDERSTANDING>
        <KEY_CHALLENGES>...</KEY_CHALLENGES>
        <SEARCH_STRATEGY>...</SEARCH_STRATEGY>
        <VALUE_CONSIDERATIONS>...</VALUE_CONSIDERATIONS>
      </THINKING>
      <CONTENT_GENERATION>
        <MAIN_SOLUTION>...</MAIN_SOLUTION>
        <TEAM_COLLABORATION_POINTS>...</TEAM_COLLABORATION_POINTS>
        <STAKEHOLDER_IMPACT>...</STAKEHOLDER_IMPACT>
      </CONTENT_GENERATION>
      <KNOWLEDGE_POINTS>
        <KNOWLEDGE_POINT>
          <TITLE>...</TITLE>
          <CONTENT>...</CONTENT>
          <TYPE>insight/fact/method/risk/recommendation</TYPE>
          <IMPORTANCE>high/medium/low</IMPORTANCE>
          <SOURCE>...</SOURCE>
        </KNOWLEDGE_POINT>
        ...
      </KNOWLEDGE_POINTS>
    </GENERATION_RESULT>
    
    Args:
        text: 包含 XML 标签的文本
        
    Returns:
        包含解析结果的字典:
        {
            'thinking': {
                'context_understanding': str,
                'key_challenges': str,
                'search_strategy': str,
                'value_considerations': str
            },
            'content': {
                'main_solution': str,
                'team_collaboration': str,
                'stakeholder_impact': str
            },
            'knowledge_points': [
                {
                    'title': str,
                    'content': str,
                    'type': str,
                    'importance': str,
                    'source': str
                },
                ...
            ]
        }
    """
    result = {
        'thinking': {},
        'content': {},
        'knowledge_points': []
    }
    
    # 提取 THINKING 部分
    thinking_block = extract_xml_tag_content(text, "THINKING")
    if thinking_block:
        result['thinking']['context_understanding'] = extract_xml_tag_content(thinking_block, "CONTEXT_UNDERSTANDING") or ""
        result['thinking']['key_challenges'] = extract_xml_tag_content(thinking_block, "KEY_CHALLENGES") or ""
        result['thinking']['search_strategy'] = extract_xml_tag_content(thinking_block, "SEARCH_STRATEGY") or ""
        result['thinking']['value_considerations'] = extract_xml_tag_content(thinking_block, "VALUE_CONSIDERATIONS") or ""
    
    # 提取 CONTENT_GENERATION 部分
    content_block = extract_xml_tag_content(text, "CONTENT_GENERATION")
    if content_block:
        result['content']['main_solution'] = extract_xml_tag_content(content_block, "MAIN_SOLUTION") or ""
        result['content']['team_collaboration'] = extract_xml_tag_content(content_block, "TEAM_COLLABORATION_POINTS") or ""
        result['content']['stakeholder_impact'] = extract_xml_tag_content(content_block, "STAKEHOLDER_IMPACT") or ""
    
    # 提取所有 KNOWLEDGE_POINT
    knowledge_blocks = extract_all_xml_blocks(text, "KNOWLEDGE_POINT")
    for kp_block in knowledge_blocks:
        title = extract_xml_tag_content(kp_block, "TITLE")
        content = extract_xml_tag_content(kp_block, "CONTENT")
        kp_type = extract_xml_tag_content(kp_block, "TYPE")
        importance = extract_xml_tag_content(kp_block, "IMPORTANCE")
        source = extract_xml_tag_content(kp_block, "SOURCE")
        
        if title and content:
            result['knowledge_points'].append({
                'title': title,
                'content': content,
                'type': kp_type if kp_type else 'insight',
                'importance': importance if importance else 'medium',
                'source': source if source else '思考'
            })
    
    return result