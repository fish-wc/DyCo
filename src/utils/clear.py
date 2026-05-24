"""
通用工具函数，用于处理和提取LLM消息内容
"""
from typing import Any

def _extract_text_from_content_block(block: Any) -> str:
    """从多种内容块结构中提取文本内容"""
    if block is None:
        return ""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        # OpenAI 兼容格式常见字段
        for key in ("text", "content", "value"):  # 优先提取text字段
            value = block.get(key)
            if isinstance(value, str):
                return value
        # 对嵌套内容进行递归提取
        for value in block.values():
            extracted = _extract_text_from_content_block(value)
            if extracted:
                return extracted
        return ""
    # 新版OpenAI SDK中的内容块对象通常有 text 属性
    text_attr = getattr(block, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(text_attr, list):
        return "".join(_extract_text_from_content_block(item) for item in text_attr)
    return ""


def extract_message_content(message: Any) -> str:
    """统一提取LLM消息中的文本内容"""
    if message is None:
        return ""

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content
    if isinstance(reasoning_content, list):
        return "".join(_extract_text_from_content_block(item) for item in reasoning_content)

    # ChatCompletionMessage 对象
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_extract_text_from_content_block(item) for item in content)

    # 字典格式的消息
    if isinstance(message, dict):
        dict_content = message.get("content")
        if isinstance(dict_content, str):
            return dict_content
        if isinstance(dict_content, list):
            return "".join(_extract_text_from_content_block(item) for item in dict_content)

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        if isinstance(reasoning, list):
            return "".join(_extract_text_from_content_block(item) for item in reasoning)

        # 有些实现会把文本放在 message["message"] 或 message["text"] 中
        for key in ("message", "text", "data"):
            value = message.get(key)
            extracted = _extract_text_from_content_block(value)
            if extracted:
                return extracted

    # 新版OpenAI SDK中message可能有一个parsed属性
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, list):
        return "".join(_extract_text_from_content_block(item) for item in parsed)

    # 兜底: 尝试直接作为字符串返回
    if isinstance(content, (dict, list)):
        return _extract_text_from_content_block(content)

    return str(message) if not isinstance(message, str) else message