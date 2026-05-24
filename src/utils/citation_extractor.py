"""
引用提取工具
从XML格式的研究报告中提取FACT测评所需的引用三元组
"""

import re
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET


def extract_citations_from_xml(article_xml: str) -> List[Dict[str, str]]:
    """
    从XML格式的研究报告中提取引用三元组
    
    Args:
        article_xml: XML格式的研究报告内容
        
    Returns:
        引用三元组列表，每个三元组格式为：
        {
            "fact": "原文中的文本片段",
            "ref_idx": "引用编号",
            "url": "引用的URL"
        }
    """
    citations = []
    
    try:
        # 解析XML
        root = ET.fromstring(article_xml)
        
        # 提取content部分
        content_elem = root.find('.//content')
        if content_elem is None:
            return citations
            
        content_text = content_elem.text or ""
        
        # 提取references部分，建立id到url的映射
        ref_mapping = {}
        refs_elem = root.find('.//references')
        if refs_elem is not None:
            for item in refs_elem.findall('item'):
                ref_id = item.get('id', '')
                url_elem = item.find('url')
                if url_elem is not None and url_elem.text:
                    ref_mapping[ref_id] = url_elem.text.strip()
        
        # 先移除CDATA标记以便处理
        content_text = content_text.replace('<![CDATA[', '').replace(']]>', '')
        
        # 使用优化后的提取逻辑
        citations = _extract_citations_from_content(content_text, ref_mapping)
    
    except Exception as e:
        print(f"提取引用时出错: {e}")
        return citations
    
    return citations


def _clean_fact_text(text: str) -> str:
    """
    清理fact文本
    - 移除Markdown标记
    - 移除多余空白
    - 保留必要的标点符号
    """
    # 移除Markdown标题标记
    text = re.sub(r'^#+\s*', '', text)
    
    # 移除Markdown粗体/斜体标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # 移除多余的空白和换行
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def extract_citations_from_research_report(report_xml: str) -> List[Dict[str, str]]:
    """
    从完整的研究报告XML中提取所有引用
    
    该函数处理包含多个章节的研究报告
    
    Args:
        report_xml: 研究报告的XML内容（可能包含多个section）
        
    Returns:
        所有引用的三元组列表
    """
    all_citations = []
    
    try:
        # 尝试解析为完整的research_report
        root = ET.fromstring(report_xml)
        
        # 查找所有content和references
        # 支持两种结构：
        # 1. 单个 <research_report><content>...</content><references>...</references></research_report>
        # 2. 多个章节的汇总
        
        # 先尝试提取整体的content
        content_elem = root.find('.//content')
        if content_elem is not None:
            content_text = content_elem.text or ""
            
            # 提取references
            ref_mapping = {}
            refs_elem = root.find('.//references')
            if refs_elem is not None:
                for item in refs_elem.findall('item'):
                    ref_id = item.get('id', '')
                    url_elem = item.find('url')
                    if url_elem is not None and url_elem.text:
                        ref_mapping[ref_id] = url_elem.text.strip()
            
            # 提取引用
            all_citations.extend(_extract_citations_from_content(content_text, ref_mapping))
    
    except Exception as e:
        print(f"从研究报告提取引用时出错: {e}")
    
    return all_citations


def _extract_citations_from_content(content_text: str, ref_mapping: Dict[str, str]) -> List[Dict[str, str]]:
    """
    从content文本中提取引用
    
    优化策略：先移除所有cite标签得到干净文本，然后根据原始cite位置找句子边界
    
    Args:
        content_text: content部分的文本内容
        ref_mapping: 引用id到url的映射
        
    Returns:
        引用三元组列表
    """
    citations = []
    
    # 移除CDATA标记
    content_text = content_text.replace('<![CDATA[', '').replace(']]>', '')
    
    # 查找所有 <cite ref="n"/> 标签及其位置
    cite_pattern = r'<cite\s+ref="(\d+)"\s*/>'
    cite_matches = list(re.finditer(cite_pattern, content_text))
    
    # 移除所有cite标签，得到干净文本
    clean_text = re.sub(cite_pattern, '', content_text)
    
    # 对每个cite标签，找到它在原文中的位置，然后在干净文本中找对应的句子
    for match in cite_matches:
        ref_idx = match.group(1)
        cite_start = match.start()
        
        # 计算在干净文本中的对应位置
        # 需要减去前面所有cite标签的总长度
        offset = 0
        for prev_match in cite_matches:
            if prev_match.start() < cite_start:
                offset += len(prev_match.group(0))
        
        adjusted_pos = cite_start - offset
        
        # 在干净文本中向前查找句子开始位置
        sentence_start = 0
        for i in range(adjusted_pos - 1, -1, -1):
            if clean_text[i] in '。！？\n':
                sentence_start = i + 1
                break
        
        # 提取fact（从句子开始到调整后的cite位置）
        fact_raw = clean_text[sentence_start:adjusted_pos].strip()
        
        # 清理fact文本
        fact = _clean_fact_text(fact_raw)
        
        # 获取URL
        url = ref_mapping.get(ref_idx, '')
        
        if fact and url:
            citations.append({
                "fact": fact,
                "ref_idx": ref_idx,
                "url": url
            })
    
    return citations


def validate_citations(citations: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    """
    验证提取的引用是否合法
    
    Args:
        citations: 引用列表
        
    Returns:
        (是否全部合法, 错误信息列表)
    """
    errors = []
    
    for i, cite in enumerate(citations):
        # 检查必要字段
        if 'fact' not in cite or not cite['fact']:
            errors.append(f"引用{i+1}: 缺少fact字段或为空")
        
        if 'ref_idx' not in cite or not cite['ref_idx']:
            errors.append(f"引用{i+1}: 缺少ref_idx字段或为空")
        
        if 'url' not in cite or not cite['url']:
            errors.append(f"引用{i+1}: 缺少url字段或为空")
        
        # 检查URL格式
        if 'url' in cite and cite['url']:
            if not (cite['url'].startswith('http://') or cite['url'].startswith('https://')):
                errors.append(f"引用{i+1}: URL格式不正确 - {cite['url']}")
    
    return (len(errors) == 0, errors)
