"""
网络搜索工具模块
提供基于Jina API的网络搜索功能,支持多种搜索配置和过滤选项。
将会被集成到base_agent.py的system工具中,供smolagent自由调用。
"""

import os, sys
import pathlib
import re
from urllib.parse import quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List
import numpy as np
from dotenv import load_dotenv, find_dotenv

# 将项目根目录加入路径（必须在导入 src 模块之前）
work_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(work_dir))
load_dotenv(find_dotenv())

from src.utils.llm_client import tool
from src.utils.llm_client import EmbeddingClient

class JinaClient:
    """简单封装 Jina HTTP 搜索接口,按需调整 base_url 与 payload。"""

    def __init__(self, api_key: str =None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.base_url = base_url or os.getenv("JINA_SEARCH_URL") or "https://s.jina.ai/"
        self.session = requests.Session()
        # Configure retries and backoff for transient network errors
        retries = int(os.getenv("JINA_HTTP_RETRIES", "1"))
        backoff = float(os.getenv("JINA_HTTP_BACKOFF", "0.5"))
        # Include 524 (Cloudflare/Gateway timeout) in retriable statuses so adapter will retry
        status_forcelist = [429, 500, 502, 503, 504, 524]
        retry_strategy = Retry(
            total=retries,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST"],
            backoff_factor=backoff,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        # Per-request timeout (increase default to better tolerate slow Jina reader)
        self.timeout = float(os.getenv("JINA_TIMEOUT", "30"))
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}",
                                         "Accept":"application/json",
                                         "X-Engine":"browser",
                                         "X-Retain-Images":"none",
                                         "X-Remove-Selector":"header, .class, #id"}
                                        )

    def web_search(
        self,
        query: str,
        count: int,
    ) -> requests.Response:
        """
        执行网络搜索

        Args:
            query: 搜索查询字符串
            count: 返回的搜索结果数量
        Returns:
            requests.Response: Jina API的响应对象
        """
        encoded_query = quote_plus(query.strip()) # URL 编码查询字符串，可以跳过人机验证
        url = self.base_url.rstrip("/") + f"/{encoded_query}?count={count}&q=Jina+AI" # NOTE 重要改进，限制检索结果数量

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"调用 Jina API 失败: {exc}") from exc

        resp.encoding = resp.apparent_encoding or "utf-8"
        # with open("E:/Code/MBTIMAS-main/workspace/messages/jina_api_response.json", "w", encoding="utf-8") as f:
        #     f.write(resp.text)
        return resp

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    def __call__(self, url: str) -> bool:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                'Authorization': f'Bearer {self.api_key}',
                'X-Timeout': "60000",
                "X-With-Generated-Alt": "true",
            }
            response = requests.get(jina_url, headers=headers)

            if response.status_code != 200:
                print(f"Jina AI Reader Failed for {url}: {response.status_code}")
                return False
            # response_dict = response.json()
            # print("data:", response_dict['data']['content'][:100])  # 打印部分内容预览
            return True
        except Exception as e:
            print(f"WebScrapingJinaTool error: {str(e)}")
            return False



def clear_content(text: str) -> str:
    """
    整理文本内容
    Args:
        text: 原始文本内容
    Returns:
        整理后的文本内容
    """
    # 1. 截断 "References" 及其之后的内容
    split_pattern = r"^[\s#]*References\s*(?::)?\s*$"
    parts = re.split(split_pattern, text, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE)
    text = parts[0]

    # 2. 删除包含双括号引用的整行 (通常是参考文献列表)
    text = re.sub(r"^.*\[\[[^\]]*\]\].*$(\r?\n|\r)?", "", text, flags=re.MULTILINE)

    # 3. 删除内联的数字引用，如 [1](url) 或 [1,2](url)
    text = re.sub(r"\[[\d,\s\-]+\]\([^\)]*\)", "", text)

    # 4. 删除其他 Markdown 链接形式 [描述](链接)
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", "", text)

    # 5. 删除仅由项目符号或常见符号构成的行（例如: "*", "-", "***", "•","* * *" 等）
    bullet_pattern = r"^[ \t]*([*\-•\u2022\u2023\u25E6]{1,}|(\* \* \*)+)[ \t]*$(\r?\n|\r)?"
    # bullet_pattern = r"^[ \t]*[\*\-•\u2022\u2023\u25E6]+[ \t]*$(\r?\n|\r)?"
    text = re.sub(bullet_pattern, "", text, flags=re.MULTILINE)

    # 6. 删除仅由单个 ASCII 标点构成的行（避免保留有意义的单字行）
    ascii_punct_single = r"^[ \t]*[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~][ \t]*$(\r?\n|\r)?"
    text = re.sub(ascii_punct_single, "", text, flags=re.MULTILINE)

    # 7. 将连续多个空行压缩为最多两个换行（即保留一个空行作为段落分隔）
    text = re.sub(r"(\r?\n\s*){3,}", "\n\n", text)

    # 8. 去除行首尾多余空白
    text = text.strip() + "\n"

    return text


def semantic_chunk(text: str, chunk_size: int = 500, overlap: int = 64) -> List[str]:
    """
    按语义进行文本切片
    
    策略：
    1. 首先按段落分割（双换行）
    2. 对过长段落，按句子分割（句号、问号、感叹号）
    3. 对过短段落，合并到前一个chunk
    4. 保持chunk大小相对均匀（目标chunk_size字符）
    
    Args:
        text: 原始文本
        chunk_size: 目标chunk大小（字符数）
        overlap: chunk之间的重叠字符数
    
    Returns:
        文本块列表
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # 预处理：清理文本
    # text = clear_content(text)
    
    # 1. 按段落分割
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # 如果当前段落很短，直接添加到current_chunk
        if len(para) < chunk_size * 0.3:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            
            # 如果累积到足够大，保存
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                # 保留overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                else:
                    current_chunk = ""
        else:
            # 段落太长，需要按句子分割
            # 保存之前的current_chunk
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 按语义标点符号分句
            sentences = re.split(r'([。！？\n;；])', para)
            # 重新组合（保留分隔符）
            sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
            if len(sentences) > 0 and sentences[-1] == '':
                sentences.pop()
            
            temp_chunk = ""
            for sent in sentences:
                if len(temp_chunk) + len(sent) < chunk_size * 1.5:
                    temp_chunk += sent
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                        # 保留overlap
                        if overlap > 0:
                            temp_chunk = temp_chunk[-overlap:] + sent
                        else:
                            temp_chunk = sent
                    else:
                        temp_chunk = sent
            
            if temp_chunk:
                current_chunk = temp_chunk
    
    # 保存最后的chunk
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # 过滤太短的chunk（提高最小长度要求，避免元数据碎片）
    chunks = [c for c in chunks if len(c) >= 80]
    
    return chunks


def semantic_filter_content(
    content: str,
    query: str,
    top_k: int = 5,
    chunk_size: int = 500,
    overlap: int = 64,
    similarity_threshold: float = 0.3,
    save_comparison: bool = True,
    result_index: int = 0,
    logger=None
) -> str:
    """
    基于语义相似度筛选内容
    
    Args:
        content: 原始内容
        query: 搜索查询
        top_k: 返回top-k个最相关的文本块
        chunk_size: 文本块大小
        similarity_threshold: 相似度阈值（低于此值的chunk会被过滤）
        save_comparison: 是否保存筛选前后对比
        result_index: 搜索结果序号（用于对比文件命名）
        logger: 日志记录器（可选）
    
    Returns:
        筛选后的内容
    """
    if not content or not query:
        return content
    
    def log(msg, level='info'):
        """统一日志输出"""
        if logger:
            if level == 'debug':
                logger.debug(msg)
            else:
                logger.info(msg)
        else:
            print(msg)
    
    log(f"\n[语义筛选 - 结果{result_index+1}] 开始处理...")
    log(f"  查询: {query}")
    log(f"  原始长度: {len(content)} 字符")
    
    try:
        # 1. 切片
        chunks = semantic_chunk(content, chunk_size=chunk_size,overlap=overlap)
        log(f"  切片结果: {len(chunks)} 个chunks")
        
        if not chunks:
            return content[:chunk_size * 5]  # 兜底：直接截断
        
        # 2. 获取embeddings
        embedding_client = EmbeddingClient()  
        
        # 获取query的embedding
        query_embeddings = embedding_client.get_embeddings([query], logger=logger)
        if not query_embeddings:
            return '\n\n'.join(chunks[:top_k])  # 兜底
        
        query_vec = query_embeddings[0]
        
        # 批量获取chunks的embeddings
        chunk_embeddings = embedding_client.get_embeddings(chunks, logger=logger)
        
        # 3. 计算相似度（考虑长度权重，避免短chunk因关键词密度高而排名靠前）
        log(f"  计算相似度...")
        similarities = []
        for i, chunk_vec in enumerate(chunk_embeddings):
            sim = embedding_client.cosine_similarity(query_vec, chunk_vec)
            
            # 长度权重：短于100字符的chunk降权，长于200字符的chunk加权
            chunk_len = len(chunks[i])
            if chunk_len < 100:
                length_penalty = 0.8  # 降权20%
            elif chunk_len > 200:
                length_penalty = 1.1  # 加权10%
            else:
                length_penalty = 1.0  # 正常权重
            
            weighted_sim = sim * length_penalty
            similarities.append((i, weighted_sim, chunks[i]))
            log(f"    Chunk {i+1}: 相似度={sim:.4f}, 长度={chunk_len}字符, 加权后={weighted_sim:.4f}", level='debug')
        
        # 4. 筛选和排序
        # 保留所有相似度 >= 0.5 的 chunk
        filtered = [item for item in similarities if item[1] >= similarity_threshold]
        
        # 如果没有符合条件的chunk，降低到similarity_threshold * 0.6重试一次
        if not filtered:
            log(f"  未找到相似度>={similarity_threshold}的chunk，降低阈值到{similarity_threshold * 0.6}重试")
            filtered = [item for item in similarities if item[1] >= similarity_threshold * 0.6]
        
        # 如果仍然没有，使用所有chunk
        if not filtered:
            log(f"  未找到相似度>={similarity_threshold * 0.6}的chunk，保留所有chunk")
            filtered = similarities
        
        # 5. 按原始顺序重新组织（保持上下文连贯性）
        filtered.sort(key=lambda x: x[0])
        
        log(f"  最终筛选结果:")
        log(f"    保留chunks: {len(filtered)} 个")
        log(f"    保留的chunk序号: {[c[0]+1 for c in filtered]}")
        if filtered:
            log(f"    相似度范围: {min(c[1] for c in filtered):.4f} ~ {max(c[1] for c in filtered):.4f}")
        
        # 6. 组合结果为XML格式，每个chunk包装在<CHUNK>标签中
        chunk_xmls = []
        for idx, (original_idx, similarity, chunk_text) in enumerate(filtered, 1):
            # 转义XML特殊字符
            chunk_escaped = chunk_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            chunk_xml = f'''<CHUNK index="{idx}" similarity="{similarity:.4f}" original_position="{original_idx+1}">{chunk_escaped}</CHUNK>'''
            chunk_xmls.append(chunk_xml)
        
        result = '\n'.join(chunk_xmls)
        
        log(f"  筛选后长度: {len(result)} 字符")
        log(f"  压缩率: {(1 - len(result)/len(content))*100:.1f}%")
        log(f"  Token节省估算: {embedding_client.total_tokens} tokens用于筛选")
        
        # 7. 保存对比文本
        if save_comparison:
            try:
                comparison_dir = os.path.join(work_dir, "workspace", "messages", "semantic_comparison")
                os.makedirs(comparison_dir, exist_ok=True)
                
                comparison_file = os.path.join(comparison_dir, f"result_{result_index+1}_comparison.txt")
                
                with open(comparison_file, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"查询: {query}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write("【原始内容】\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"长度: {len(content)} 字符\n")
                    f.write(f"Chunks数量: {len(chunks)}\n")
                    f.write("-" * 80 + "\n")
                    f.write(content)
                    f.write("\n\n")
                    
                    f.write("【相似度分析】\n")
                    f.write("-" * 80 + "\n")
                    for i, sim, chunk in similarities:
                        status = "✓ 保留" if any(x[0] == i for x in filtered) else "✗ 过滤"
                        f.write(f"Chunk {i+1} [{status}]: 相似度={sim:.4f}, 长度={len(chunk)}字符\n")
                        f.write(f"内容: {chunk}\n\n")
                    f.write("\n")
                    
                    f.write("【筛选后内容】\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"长度: {len(result)} 字符\n")
                    f.write(f"保留Chunks: {len(filtered)} 个\n")
                    f.write(f"压缩率: {(1 - len(result)/len(content))*100:.1f}%\n")
                    f.write("-" * 80 + "\n")
                    f.write(result)
                    f.write("\n\n")
                    
                    f.write("【统计信息】\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"原始字符数: {len(content)}\n")
                    f.write(f"筛选后字符数: {len(result)}\n")
                    f.write(f"压缩率: {(1 - len(result)/len(content))*100:.1f}%\n")
                    f.write(f"Embedding Token消耗: {embedding_client.total_tokens}\n")
                    f.write(f"相似度阈值: 0.5（兜底0.3）\n")
                    f.write(f"Chunk大小: {chunk_size}\n")
                
                log(f"  对比文件已保存: {comparison_file}")
            except Exception as save_error:
                log(f"  保存对比文件失败: {save_error}")
        
        return result
        
    except Exception as e:
        # 如果embedding失败，使用兜底策略：直接截断
        error_msg = f"\n[语义筛选失败] {e}，使用兜底策略"
        if logger:
            logger.warning(error_msg)
        else:
            print(error_msg)
        return content[:chunk_size * top_k]


@tool
def web_search_tool(
    search_query: str,count: int = 1,max_retries: int = 1, logger=None
) -> str:
    """
    网络搜索工具，使用Jina API进行搜索并返回结构化结果
    
    Args:
        search_query: 搜索查询字符串
        count: 返回结果数量
        max_retries: 最大重试次数
        logger: 日志记录器
    
    Returns:
        搜索结果的JSON字符串
    """
    """
    网络搜索工具函数
    Args:
        search_query: 搜索查询字符串
        count: 返回的搜索结果数量。考虑到模型上下文限制，最好设置为1。
        max_retries: 最大重试次数，不需要用户传入，仅供内部调用时使用。
        logger: 日志记录器（可选），用于统一日志输出
    Returns:
        搜索结果的字符串表示
    """
    # NOTE 考虑到token消耗太多了，这里优化一下请求数量.2025.12.26
    # more_count = max(int(count * 1.2), 5)  # 多请求一些结果以便过滤无效结果
    more_count = count
    
    def log(msg, level='info'):
        """统一日志输出"""
        if logger:
            if level == 'debug':
                logger.debug(msg)
            elif level == 'warning':
                logger.warning(msg)
            else:
                logger.info(msg)
        else:
            print(msg) 

    try:
        log(f"[Web Search] 开始搜索: {search_query}")
        log(f"  请求结果数: {more_count}", level='debug')
        
        jina_search = JinaClient()
        response = jina_search.web_search(
            query=search_query,
            count=more_count,
        )

        # 解析响应数据
        response_data = response.json()
        log(f"  收到API响应，开始解析...", level='debug')
        
        # 构建 XML 格式的返回结果
        xml_results = "<SEARCH_RESULTS>\n"
        xml_results += f"    <QUERY>{search_query}</QUERY>\n"
        
        # 提取 token 使用信息
        total_tokens = 0
        valid_results = 0
        
        # 遍历 response_data["data"] 并构建 XML 结果
        for _, item in enumerate(response_data["data"], start=1):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            
            # 过滤无效结果
            if not (title or url or content):
                continue
            if "Are you a robot?" in content or "Security check required" in content:
                continue
            
            # 提取描述字段（在处理content之前）
            description = item.get("description", "")
            
            # 从content中移除description部分（避免重复切片）
            if description and description.strip() and len(content) > len(description) * 1.5:
                # 尝试从content开头移除description（Jina API通常将description放在content开头）
                content_without_desc = content
                removed = False
                
                # 策略1：直接匹配（content以description开头）
                if content.strip().startswith(description.strip()):
                    content_without_desc = content[len(description):].strip()
                    removed = True
                    log(f"  结果{valid_results+1}: 使用直接匹配移除description", level='debug')
                else:
                    # 策略2：模糊匹配（处理"摘要:""Abstract:"等前缀）
                    # 在content中查找description的位置（允许一定偏移）
                    desc_stripped = description.strip()
                    # 检查前200个字符中是否包含description
                    search_range = content[:min(len(content), len(desc_stripped) + 200)]
                    
                    # 尝试找到description在content中的精确位置
                    idx = search_range.find(desc_stripped)
                    if idx != -1 and idx < 50:  # 允许前面最多50个字符的前缀
                        # 找到了，移除从开头到description结尾的部分
                        content_without_desc = content[idx + len(desc_stripped):].strip()
                        removed = True
                        log(f"  结果{valid_results+1}: 使用模糊匹配移除description（偏移{idx}字符）", level='debug')
                    else:
                        # 策略3：标准化匹配（处理空格差异）
                        desc_normalized = ' '.join(description.split())
                        content_normalized = ' '.join(content.split())
                        if content_normalized.startswith(desc_normalized):
                            # 计算原始content中description的实际长度
                            desc_len = 0
                            temp_content = content
                            for word in description.split():
                                idx = temp_content.find(word)
                                if idx != -1:
                                    desc_len = idx + len(word)
                                    temp_content = temp_content[idx + len(word):]
                                else:
                                    break
                            if desc_len > 0:
                                content_without_desc = content[desc_len:].strip()
                                removed = True
                                log(f"  结果{valid_results+1}: 使用标准化匹配移除description", level='debug')
                
                # 只有在移除后content仍然足够长时才使用
                if removed and len(content_without_desc) > 100:
                    content = content_without_desc
                    log(f"  移除成功: 剩余{len(content)}字符", level='debug')
                else:
                    if removed:
                        log(f"  结果{valid_results+1}: description移除后内容过短（{len(content_without_desc)}字符），保留原content", level='debug')
                    else:
                        log(f"  结果{valid_results+1}: 未找到匹配的description，保留原content", level='debug')
            
            # 使用语义筛选处理content（基于search_query的相关性）
            # 返回的content已经是包含<CHUNK>标签的XML格式
            try:
                similarity_threshold = 0.4  # 可调整阈值以控制筛选严格度
                content = semantic_filter_content(
                    content=content,
                    query=search_query,
                    top_k=5,
                    chunk_size=628,
                    similarity_threshold=similarity_threshold,
                    save_comparison=True,
                    result_index=valid_results,
                    logger=logger
                ) # TODO chunk_size 可以放到配置文件中
            except Exception as e:
                # 兜底：简单截断，也包装为CHUNK格式
                error_msg = f"\n[语义筛选出错] {e}，使用简单截断"
                if logger:
                    logger.warning(error_msg)
                else:
                    print(error_msg)
                truncated = clear_content(content)[:30000]
                content_escaped = truncated.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                content = f'<CHUNK index="1" similarity="0.0000" original_position="1">{content_escaped}</CHUNK>'
            
            # 检查content是否为空（语义筛选后可能没有返回任何CHUNK）
            if not content or not content.strip() or '<CHUNK' not in content:
                log(f"  结果{valid_results+1}: content为空或无有效CHUNK，跳过此结果", level='debug')
                continue
            
            # 提取其他字段
            date = item.get("date", "")
            metadata = item.get("metadata", {})
            keywords = metadata.get("keywords", "")
            
            # 统计 token 使用
            item_usage = item.get("usage", {})
            total_tokens += item_usage.get("tokens", 0)
            
            # 构建 XML RESULT 块
            valid_results += 1
            xml_results += f"    <RESULT index=\"{valid_results}\">\n"
            
            if title:
                title_escaped = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                xml_results += f"        <TITLE>{title_escaped}</TITLE>\n"
            if url:
                xml_results += f"        <URL>{url}</URL>\n"
            if date:
                xml_results += f"        <DATE>{date}</DATE>\n"
            if description:
                desc_escaped = description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                xml_results += f"        <DESCRIPTION>{desc_escaped}</DESCRIPTION>\n"
            if keywords:
                # keywords 可能是字符串或列表，统一转换为字符串后再转义
                try:
                    if isinstance(keywords, list):
                        kw_str = ", ".join(str(k) for k in keywords)
                    else:
                        kw_str = str(keywords)
                except Exception:
                    kw_str = str(keywords)

                keywords_escaped = kw_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                xml_results += f"        <KEYWORDS>{keywords_escaped}</KEYWORDS>\n"
            if content:
                # content已经包含<CHUNK>标签，直接嵌入（不需要再转义）
                xml_results += f"        <CONTENT>\n{content}\n        </CONTENT>\n"
            
            xml_results += f"    </RESULT>\n"
        
        # 如果没有有效结果，重试或返回空结果
        if valid_results == 0:
            if max_retries > 0:
                log(f"  未找到有效结果，重试中... (剩余重试次数: {max_retries})")
                return web_search_tool(search_query, count, max_retries - 1, logger)
            else:
                log("  搜索完成：未找到有效结果", level='warning')
                return f"""<SEARCH_RESULTS>\n    <QUERY>{search_query}</QUERY>\n    <STATUS>未找到有效结果</STATUS>\n    <MESSAGE>已尝试多次过滤仍无有效结果</MESSAGE>\n</SEARCH_RESULTS>"""
        
        # 添加统计信息
        xml_results = xml_results[:xml_results.find('<RESULT')] + \
                      f"    <TOTAL_RESULTS>{valid_results}</TOTAL_RESULTS>\n" + \
                      (f"    <TOKEN_USAGE>\n        <TOTAL>{total_tokens}</TOTAL>\n    </TOKEN_USAGE>\n" if total_tokens > 0 else "") + \
                      xml_results[xml_results.find('<RESULT'):]
        
        xml_results += "</SEARCH_RESULTS>"
        
        log(f"[Web Search] 搜索完成")
        log(f"  有效结果数: {valid_results}")
        log(f"  Token消耗: {total_tokens}")
        
        return xml_results
    except Exception as e:
        error_msg = f"搜索失败: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        else:
            print(error_msg)
        return error_msg




if __name__ == "__main__":
    # 简单测试
    result = web_search_tool("人工智能之父", count=3)
    path = r"E:\Code\MBTIMAS-main\workspace\messages\jina_search_result.txt"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)
        
