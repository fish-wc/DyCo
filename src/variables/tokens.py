'''
设定tokens记录的全局变量和函数
'''
# 初始化或复用全局 Token 统计器
if "TOKEN_STATS" not in globals():
    TOKEN_STATS = {
        "prompt": 0,
        "completion": 0,
        "total": 0,
        "history": [],
        # 分类统计
        "by_category": {
            "llm": {"prompt": 0, "completion": 0, "total": 0, "count": 0},
            "embedding": {"prompt": 0, "completion": 0, "total": 0, "count": 0},
            "search": {"prompt": 0, "completion": 0, "total": 0, "count": 0},
        }
    }

def record_token_usage(prompt_tokens=0, completion_tokens=0, total_tokens=None, label=None, category="llm"):
    """
    将一次调用的 token 结果累计到全局 TOKEN_STATS
    
    Args:
        prompt_tokens: 输入token数
        completion_tokens: 输出token数
        total_tokens: 总token数（可选，默认为prompt+completion）
        label: 调用标签
        category: token类别（'llm', 'embedding', 'search'）
    """
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
    label = label or f"调用 {len(TOKEN_STATS['history']) + 1}"
    
    # 累计到总计
    TOKEN_STATS["prompt"] += prompt_tokens
    TOKEN_STATS["completion"] += completion_tokens
    TOKEN_STATS["total"] += total_tokens
    
    # 累计到分类统计
    if category in TOKEN_STATS["by_category"]:
        TOKEN_STATS["by_category"][category]["prompt"] += prompt_tokens
        TOKEN_STATS["by_category"][category]["completion"] += completion_tokens
        TOKEN_STATS["by_category"][category]["total"] += total_tokens
        TOKEN_STATS["by_category"][category]["count"] += 1
    
    entry = {
        "label": label,
        "category": category,
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": total_tokens,
    }
    TOKEN_STATS["history"].append(entry)
    return entry

# TODO 把所有调用处都改成用这个函数
def record_token_usage_from_resp(resp):
    prompt_tokens = 0
    completion_tokens = 0
    usage = getattr(resp, 'usage', None)
    if usage is not None:
        if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens
    try:
        record_token_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    except Exception:
        print("记录 Token 使用情况时出错")
        pass


def get_current_token_cost():
    """获取当前累计的 Token 统计数据"""
    return TOKEN_STATS["history"][-1] if TOKEN_STATS["history"] else None

def get_total_token_usage():
    """获取累计的 Token 使用情况"""
    return {
        "prompt": TOKEN_STATS["prompt"],
        "completion": TOKEN_STATS["completion"],
        "total": TOKEN_STATS["total"],
    }

def reset_token_stats():
    """清空累计数据"""
    TOKEN_STATS["prompt"] = 0
    TOKEN_STATS["completion"] = 0
    TOKEN_STATS["total"] = 0
    TOKEN_STATS["history"].clear()
    
COST_DICT = {
    "gpt-4o-mini": {
        "prompt": 0.03 / 1000000,
        "completion": 0.06 / 1000000,
    },
}