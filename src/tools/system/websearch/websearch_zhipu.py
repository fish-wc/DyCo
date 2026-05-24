"""
网络搜索工具模块
提供基于智谱AI的网络搜索功能,支持多种搜索配置和过滤选项。
将会被集成到base_agent.py的system工具中,供smolagent自由调用。
"""

import os, sys
import pathlib
work_dir=pathlib.Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(work_dir))

from typing import Optional, Dict, Any, Literal
from dotenv import load_dotenv, find_dotenv
from zai import ZhipuAiClient
from smolagents import tool
from src.prompts import prompt_loader

# 加载环境变量
load_dotenv(find_dotenv())


class WebSearchManager:
    """网络搜索管理器,封装智谱AI的网络搜索功能"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化网络搜索管理器
        
        Args:
            api_key: 智谱AI的API密钥,如果为None则从环境变量ZHIPU_API_KEY读取
            logger: 日志记录器
        """
        self.api_key = api_key or os.getenv("ZHIMU_API_KEY")
        if not self.api_key:
            raise ValueError("未找到智谱AI的API密钥,请设置环境变量ZHIPU_API_KEY或传入api_key参数")
        
        self.client = ZhipuAiClient(api_key=self.api_key)
        
    
    def search(
        self,
        search_query: str,
        search_engine: Literal["search_pro", "search_lite"] = "search_pro",
        count: int = 10,
        search_domain_filter: Optional[str] = None,
        search_recency_filter: Literal["noLimit", "day", "week", "month", "year"] = "noLimit",
        content_size: Literal["low", "medium", "high"] = "medium"
    ) -> Dict[str, Any]:
        """
        执行网络搜索
        
        Args:
            search_query: 搜索查询字符串
            search_engine: 搜索引擎类型,可选 "search_pro" 或 "search_lite"
            count: 返回结果的条数,范围1-50,默认10
            search_domain_filter: 只访问指定域名的内容,例如 "www.sohu.com"
            search_recency_filter: 搜索指定日期范围内的内容
                - noLimit: 不限制时间(默认)
                - day: 最近一天
                - week: 最近一周
                - month: 最近一个月
                - year: 最近一年
            content_size: 控制网页摘要的字数
                - low: 少量文本
                - medium: 中等文本(默认)
                - high: 大量文本
        
        Returns:
            搜索结果字典,包含搜索到的网页信息
        """
        try:
            # 参数校验
            if not search_query or not search_query.strip():
                raise ValueError("搜索查询不能为空")
            
            if not 1 <= count <= 50:
                print(f"count参数超出范围[1-50],已调整为: {min(max(count, 1), 50)}")
                count = min(max(count, 1), 50)
            
            # 执行搜索
            response = self.client.web_search.web_search(
                search_engine=search_engine,
                search_query=search_query,
                count=count,
                search_domain_filter=search_domain_filter,
                search_recency_filter=search_recency_filter,
                content_size=content_size
            )
            search_results = response.search_result
            # print("search_results:", search_results)
            template = prompt_loader.load_system_prompt("_websearch") # 加载搜索结果格式化模板
            result_formatted = ""
            for idx, result in enumerate(search_results):
                if result.link is None:
                    continue  # 跳过无效结果
                result_cur = template.format(
                    idx=idx+1,
                    publish_date=result.publish_date,
                    media=result.media,
                    title=result.title,
                    link=result.link,
                    content=result.content
                )
                result_formatted += result_cur + "\n"

            return result_formatted
            
        except Exception as e:
            print(f"网络搜索失败: {str(e)}")
            raise


# 全局搜索管理器实例
_search_manager = None


def get_search_manager() -> WebSearchManager:
    """获取全局搜索管理器实例"""
    global _search_manager
    if _search_manager is None:
        _search_manager = WebSearchManager()
    return _search_manager


@tool
def web_search_tool(
    search_query: str,
    # count: int = 3,
    search_domain_filter: Optional[str] = None,
    search_recency_filter: str = "noLimit",
    content_size: str = "low",
) -> str:
    """
    执行网络搜索并返回结果。使用智谱AI的搜索引擎获取最新的网络信息。
    
    Args:
        search_query: 搜索查询字符串,描述你想要搜索的内容
        search_domain_filter: 只搜索指定域名的内容,例如 "www.sohu.com",不指定则搜索全网
        search_recency_filter: 时间范围过滤,可选值: "noLimit"(不限制), "day"(最近一天), "week"(最近一周), "month"(最近一个月), "year"(最近一年)
        content_size: 摘要详细程度,可选值: "low"(少量文本), "medium"(中等文本), "high"(大量文本),默认"high"
    Returns:
        搜索结果的字符串表示,包含网页标题、链接和摘要等信息
    """
    count = 2  # 固定返回5条结果
    try:
        manager = get_search_manager()
        response = manager.search(
            search_query=search_query,
            search_engine="search_pro",
            count=count,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
            content_size=content_size
        )
        
        return response
        
    except Exception as e:
        return f"搜索失败: {str(e)}"


if __name__ == "__main__":
    # 简单测试
    result = web_search_tool("一文看透中国居民收入分布")
    print("搜索结果:")
    print(result)