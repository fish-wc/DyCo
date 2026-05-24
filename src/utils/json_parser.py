"""
JSON 解析工具
用于从 LLM 返回的文本中提取和解析 JSON 数据
"""
import json
import re
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """JSON 解析错误"""
    pass


def extract_json_from_text(text: str, strict: bool = False) -> Optional[Dict[str, Any]]:
    """
    从文本中提取并解析 JSON
    
    支持多种格式:
    1. Markdown 代码块包裹的 JSON: ```json {...} ```
    2. 纯 JSON 文本
    3. 混合文本中的 JSON (在 { 和 } 之间)
    
    Args:
        text: 包含 JSON 的文本
        strict: 是否使用严格模式(如果为True,解析失败会抛出异常)
        
    Returns:
        解析后的 JSON 字典,如果解析失败返回 None
        
    Raises:
        JSONParseError: 当 strict=True 且解析失败时
    """
    if not text or not isinstance(text, str):
        if strict:
            raise JSONParseError(f"输入文本无效: {type(text)}")
        return None
    
    # 策略1: 尝试从 Markdown 代码块中提取
    json_str = _extract_from_markdown_block(text)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"从 Markdown 代码块解析失败: {e}")
    
    # 策略2: 尝试直接解析整个文本
    try:
        text_stripped = text.strip()
        if text_stripped.startswith('{') and text_stripped.endswith('}'):
            return json.loads(text_stripped)
    except json.JSONDecodeError as e:
        logger.debug(f"直接解析失败: {e}")
    
    # 策略3: 寻找第一个完整的 JSON 对象
    json_str = _extract_first_json_object(text)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"提取 JSON 对象解析失败: {e}")
    
    # 所有策略都失败
    if strict:
        raise JSONParseError(f"无法从文本中提取有效的 JSON:\n{text[:200]}...")
    
    logger.warning(f"JSON 解析失败,返回 None")
    return None


def _extract_from_markdown_block(text: str) -> Optional[str]:
    """
    从 Markdown 代码块中提取 JSON
    
    支持格式:
    - ```json {...} ```
    - ```{...}```
    
    Args:
        text: 输入文本
        
    Returns:
        提取的 JSON 字符串,如果未找到返回 None
    """
    # 匹配 ```json ... ``` 或 ``` ... ```
    patterns = [
        r'```json\s*(\{.*?\})\s*```',  # ```json {...} ```
        r'```\s*(\{.*?\})\s*```',       # ```{...}```
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            logger.debug(f"从 Markdown 代码块中提取到 JSON (模式: {pattern[:20]}...)")
            return match.group(1)
    
    return None


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    从文本中提取第一个完整的 JSON 对象
    
    通过寻找匹配的 { 和 } 来定位 JSON 对象
    
    Args:
        text: 输入文本
        
    Returns:
        提取的 JSON 字符串,如果未找到返回 None
    """
    # 寻找第一个 {
    start = text.find('{')
    if start == -1:
        return None
    
    # 从 start 位置开始,寻找匹配的 }
    bracket_count = 0
    end = -1
    
    for i in range(start, len(text)):
        if text[i] == '{':
            bracket_count += 1
        elif text[i] == '}':
            bracket_count -= 1
            if bracket_count == 0:
                end = i
                break
    
    if end == -1:
        # 如果没找到匹配的右括号,使用最后一个 }
        end = text.rfind('}')
        if end == -1 or end < start:
            return None
    
    json_str = text[start:end+1]
    logger.debug(f"提取到 JSON 对象,长度: {len(json_str)} 字符")
    return json_str


def extract_json_array_from_text(text: str, strict: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    从文本中提取并解析 JSON 数组
    
    Args:
        text: 包含 JSON 数组的文本
        strict: 是否使用严格模式
        
    Returns:
        解析后的 JSON 数组,如果解析失败返回 None
        
    Raises:
        JSONParseError: 当 strict=True 且解析失败时
    """
    if not text or not isinstance(text, str):
        if strict:
            raise JSONParseError(f"输入文本无效: {type(text)}")
        return None
    
    # 尝试从 Markdown 代码块中提取
    patterns = [
        r'```json\s*(\[.*?\])\s*```',  # ```json [...] ```
        r'```\s*(\[.*?\])\s*```',       # ```[...]```
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    # 尝试直接解析
    try:
        text_stripped = text.strip()
        if text_stripped.startswith('[') and text_stripped.endswith(']'):
            return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass
    
    # 提取第一个数组
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    if strict:
        raise JSONParseError(f"无法从文本中提取有效的 JSON 数组:\n{text[:200]}...")
    
    return None


def safe_parse_json(text: str, default: Any = None) -> Any:
    """
    安全地解析 JSON,如果失败返回默认值
    
    Args:
        text: JSON 文本
        default: 解析失败时的默认返回值
        
    Returns:
        解析结果或默认值
    """
    try:
        result = extract_json_from_text(text, strict=False)
        return result if result is not None else default
    except Exception as e:
        logger.debug(f"JSON 解析出错: {e}")
        return default
