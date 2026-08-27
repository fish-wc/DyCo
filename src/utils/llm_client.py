"""
大模型使用集成
"""
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from typing import Any, Optional

from openai import OpenAI
try:
    from zai import ZhipuAiClient
except ImportError:
    try:
        # 部分版本的 zai-sdk 未在顶层导出 ZhipuAiClient
        from zai._client import ZhipuAiClient
    except ImportError:
        class ZhipuAiClient:  # zai-sdk 未安装时的占位，保证 isinstance 检查不报错
            pass
import requests
import numpy as np


import inspect
import json
import time
from typing import Callable, Dict, List, Union
from functools import wraps


from .clear import extract_message_content
from src.variables.tokens import record_token_usage_from_resp
from src.prompts import prompt_loader

def create_llm_client(config:Optional[Any]=None,
                      use_smolagent: bool =False,
                      embedding: bool = False,
                      ) -> Any:
    """
    根据配置创建 LLM 客户端
    
    Args:
        config: 模型配置对象。
        use_smolagent: 是否使用 smolagents 兼容模式 (默认False)
        embedding: 是否为嵌入模型 (默认False)
        
    Returns:
        LLM 客户端（OpenAI）
    """

    if use_smolagent and embedding:
        # 报错，使用方式错误
        raise ValueError("use_smolagent 和 embedding 不能同时为 True")
    
    # 用 openai 使用embedding
    if embedding:
        # 如果使用的是embedding模式，那么config必定为system_config TODO 这里可以统一为ModelConfig模式
        if not hasattr(config, "embedding"):
            raise ValueError("embedding 模式下必须提供系统配置对象作为 config 参数")
        embedding_config= config.embedding
        embedding_api_key = os.getenv(embedding_config.api_key)
        embedding_model_url = os.getenv(embedding_config.model_url)
        
        client = OpenAI(api_key=embedding_api_key,
                        base_url=embedding_model_url,)
        return client
    
    # 获取环境变量中的 API Key 和模型URL
    api_key = os.getenv(config.api_key)
    model_url = os.getenv(config.model_url)
    
    if use_smolagent:
        try:
            from smolagents import OpenAIServerModel
            return OpenAIServerModel(
                model_id=config.model_name,
                api_base=model_url,
                api_key=api_key,
            )
        except ImportError:
            raise ImportError("需要安装 smolagents 库以使用 smolagents 兼容模式: pip install smolagents")

    if "bigmodel" in model_url: # 说明是智谱清言的api，单独测试一下不采用深度思考的方式
        try:
            print(f"使用 智谱清言 客户端: {model_url}")
            return ZhipuAiClient(
                api_key=api_key,
                base_url=model_url,
                timeout=config.timeout,
            )
        except ImportError:
            raise ImportError("")
    else:
        # 使用 OpenAI 客户端
        try:
            return OpenAI(
                api_key=api_key,
                base_url=model_url,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("需要安装 openai 库: pip install openai")

def call_llm(
    llm_client,
    model_name: str,
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    logger=None,
    logger_name: str = ""
) -> str:
    """
    通用的LLM调用方法
    
    Args:
        llm_client: LLM客户端实例
        model_name: 模型名称
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        temperature: 温度参数
        max_tokens: 最大token数
        logger: 日志记录器
        logger_name: 日志记录名称
        
    Returns:
        LLM生成的文本
    """
    if logger:
        logger.debug("调用LLM进行文本生成...")
        logger.debug(f"  模型: {model_name}")
        logger.debug(f"  温度: {temperature}")
        logger.debug(f"  最大tokens: {max_tokens}")
    
    if not system_prompt:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
    try:
        if isinstance(llm_client, ZhipuAiClient):
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                thinking={
                    "type": "disabled",    # 不启用深度思考模式
                },
                temperature=temperature,
                max_tokens=max_tokens
            )
            message_obj = response.choices[0].message
            response_text = extract_message_content(message_obj)
            
            if logger: # 考虑到这里报错后文检查不到，添加一个日志。
                logger.debug(f"大模型返回字数：{len(response_text)}")
            if not response_text:
                if logger:
                    logger.warning("LLM返回内容为空，记录原始消息: %s", message_obj)
                response_text = str(message_obj)
        else:
            # OpenAI客户端
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            message_obj = response.choices[0].message
            response_text = extract_message_content(message_obj)
            
            if logger: # 考虑到这里报错后文检查不到，添加一个日志。
                logger.debug(f"大模型返回字数：{len(response_text)}\n response: {response}")
            if not response_text:
                if logger:
                    logger.warning("LLM返回内容为空，记录原始消息: %s", message_obj)
                response_text = str(message_obj)
                
        # 记录token使用（分类为llm）
        try:
            from src.variables.tokens import record_token_usage
            usage = getattr(response, 'usage', None)
            if usage is not None:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
                record_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    label=logger_name or "LLM调用",
                    category="llm"
                )
        except Exception:
            # 降级使用原有方法
            record_token_usage_from_resp(response)
        
        if logger:
            logger.debug("\n 【大模型生成开始】"+"="*180)
            logger.info(f"任务完成：{logger_name}")
            logger.debug(f"【SYSTEM PROMPT】\n{system_prompt}") 
            logger.debug(f"【USER PROMPT】\n{user_prompt}") 
            logger.debug(f"【LLM RESULT】\n{response_text}")
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0) 
                logger.debug(f"【TOKEN USAGE】 Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")
            logger.debug("\n 【大模型生成结束】"+"="*180)
        
        return response_text
        
    except Exception as e:
        if logger:
            logger.error(f"LLM调用失败: {e}", exc_info=True)
        raise


# ========== 新增：通用 LLM 工具调用类 ==========


# ========== 全局工具注册装饰器（类似 smolagents）==========

def tool(func: Callable = None, *, name: str = None, description: str = None) -> Callable:
    """
    全局工具注册装饰器（模仿 smolagents 的设计）
    
    可以在任何地方使用此装饰器注册工具，然后将工具列表传给 LLMToolClient
    
    使用方式1（无参数）:
        @tool
        def my_tool(query: str) -> str:
            '''工具描述'''
            return result
    
    使用方式2（指定名称和描述）:
        @tool(name="custom_name", description="自定义描述")
        def my_tool(query: str) -> str:
            return result
    
    Args:
        func: 工具函数
        name: 工具名称（可选，默认使用函数名）
        description: 工具描述（可选，默认使用文档字符串）
        
    Returns:
        装饰后的函数，附带工具元数据
    """
    def decorator(f: Callable) -> Callable:
        # 保存工具元数据到函数对象
        f._is_tool = True
        f._tool_name = name or f.__name__
        f._tool_description = description or (f.__doc__ or "").strip()
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # 复制元数据到 wrapper
        wrapper._is_tool = True
        wrapper._tool_name = f._tool_name
        wrapper._tool_description = f._tool_description
        
        return wrapper
    
    # 支持 @tool 和 @tool() 两种用法
    if func is None:
        # 被调用为 @tool(name="xxx")
        return decorator
    else:
        # 被调用为 @tool
        return decorator(func)


def is_tool(func: Callable) -> bool:
    """
    检查函数是否是已注册的工具
    
    Args:
        func: 待检查的函数
        
    Returns:
        是否为工具
    """
    return hasattr(func, '_is_tool') and func._is_tool


class LLMToolClient:
    """
    大模型工具调用通用类
    
    支持功能：
    1. 通过装饰器注册工具函数
    2. 自动从函数签名和文档字符串生成工具描述
    3. 支持新旧版本 OpenAI API（tools 和 functions）
    4. 处理工具调用的完整流程
    5. 兼容不同大模型（OpenAI、ZhipuAI等）
    
    使用示例：
        # 方式1: 使用全局装饰器预注册工具
        from src.utils.llm_client import tool, LLMToolClient
        
        @tool
        def search_web(query: str, count: int = 3) -> str:
            '''搜索网页'''
            return "搜索结果"
        
        @tool(name="calc", description="计算器")
        def calculator(expr: str) -> str:
            return str(eval(expr))
        
        # 传入工具列表创建客户端
        client = LLMToolClient(config, tools=[search_web, calculator])
        
        # 方式2: 实例化后注册工具（向后兼容）
        client = LLMToolClient(config)
        
        @client.register_tool()
        def another_tool(arg: str) -> str:
            return "result"
        
        # 调用
        response = client.call_with_tools(
            messages=[{"role": "user", "content": "帮我搜索Python教程"}],
            model_name="gpt-4"
        )
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        tools: Optional[List[Callable]] = None,
        use_smolagent: bool = False,
        embedding: bool = False,
        use_tools_api: bool = True,
        logger = None,
        enable_compatibility_mode: bool = True,
        force_compatibility_mode: bool = True
    ):
        """
        初始化 LLM 工具客户端
        
        Args:
            config: 模型配置对象
            tools: 预注册的工具函数列表（使用 @tool 装饰器注册的函数）
            use_smolagent: 是否使用 smolagents 兼容模式
            embedding: 是否为嵌入模型
            use_tools_api: True使用新版tools API，False使用旧版functions API
            logger: 日志记录器
            enable_compatibility_mode: 是否启用兼容模式（当原生工具调用失败时自动切换）
            force_compatibility_mode: 是否强制使用兼容模式（绕过原生API，直接使用提示词工程）
        """
        self.config = config
        self.use_tools_api = use_tools_api
        self.logger = logger
        self.enable_compatibility_mode = enable_compatibility_mode
        self.force_compatibility_mode = force_compatibility_mode
        
        # 创建底层 LLM 客户端
        self.client = create_llm_client(config, use_smolagent, embedding)
        
        # 工具注册表：{tool_name: {func, description, schema}}
        self._tools: Dict[str, Dict] = {}
        
        # 工具调用模式：'native' 或 'compatibility'
        self._tool_call_mode = 'compatibility' if force_compatibility_mode else 'native'
        
        # 注册传入的工具列表
        if tools:
            self._register_tools_from_list(tools)
        
        if self.logger:
            if force_compatibility_mode:
                self.logger.info("🔧 强制启用工具调用兼容模式（提示词工程）")
            elif enable_compatibility_mode:
                self.logger.info("🔧 工具调用兼容模式已启用（自动回退）")
        
    def _register_tools_from_list(self, tools: List[Callable]) -> None:
        """
        从工具列表批量注册工具
        
        Args:
            tools: 工具函数列表（应使用 @tool 装饰器装饰）
        """
        for tool_func in tools:
            if not callable(tool_func):
                if self.logger:
                    self.logger.warning(f"跳过非可调用对象: {tool_func}")
                continue
            
            # 检查是否是用 @tool 装饰的函数
            if is_tool(tool_func):
                tool_name = tool_func._tool_name
                tool_description = tool_func._tool_description
            else:
                # 如果没有使用 @tool，也允许注册，使用函数名和文档字符串
                tool_name = tool_func.__name__
                tool_description = (tool_func.__doc__ or "").strip()
                if self.logger:
                    self.logger.debug(f"工具 {tool_name} 未使用 @tool 装饰器，使用默认元数据")
            
            # 生成 schema
            schema = self._generate_schema(tool_func, tool_description)
            
            # 注册工具
            self._tools[tool_name] = {
                "func": tool_func,
                "description": tool_description,
                "schema": schema
            }
            
            if self.logger:
                self.logger.debug(f"从列表注册工具: {tool_name}")
        
    def register_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """
        装饰器：注册工具函数
        
        Args:
            name: 工具名称（默认使用函数名）
            description: 工具描述（默认从函数文档字符串提取）
            
        Returns:
            装饰后的函数
            
        使用示例：
            @client.register_tool()
            def my_tool(arg1: str, arg2: int = 10) -> str:
                '''工具描述'''
                return result
                
            # 或指定名称和描述
            @client.register_tool(name="custom_name", description="自定义描述")
            def my_tool(arg1: str) -> str:
                return result
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or (func.__doc__ or "").strip()
            
            # 从函数签名生成参数 schema
            schema = self._generate_schema(func, tool_description)
            
            # 注册工具
            self._tools[tool_name] = {
                "func": func,
                "description": tool_description,
                "schema": schema
            }
            
            if self.logger:
                self.logger.debug(f"注册工具: {tool_name}")
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def _generate_schema(self, func: Callable, description: str) -> Dict:
        """
        从函数签名生成 OpenAI 工具参数 schema
        
        Args:
            func: 函数对象
            description: 函数描述
            
        Returns:
            参数 schema 字典
        """
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # 跳过 self 和特殊参数
            if param_name in ('self', 'cls'):
                continue
                
            # 获取参数类型
            param_type = "string"  # 默认
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list or param.annotation == List:
                    param_type = "array"
                elif param.annotation == dict or param.annotation == Dict:
                    param_type = "object"
            
            # 从文档字符串提取参数描述
            param_desc = f"参数 {param_name}"
            if func.__doc__:
                # 简单的文档字符串解析
                for line in func.__doc__.split('\n'):
                    if param_name in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            param_desc = parts[1].strip()
                        break
            
            properties[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            
            # 如果参数没有默认值，则为必需参数
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
            
        return schema
    
    def _build_tool_definitions(self) -> List[Dict]:
        """
        构建工具定义列表（适配新旧版 API）
        
        Returns:
            工具定义列表
        """
        if self.use_tools_api:
            # 新版 tools API 格式
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info["description"],
                        "parameters": tool_info["schema"]
                    }
                }
                for tool_name, tool_info in self._tools.items()
            ]
        else:
            # 旧版 functions API 格式
            return [
                {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": tool_info["schema"]
                }
                for tool_name, tool_info in self._tools.items()
            ]
    
    def _build_compatibility_system_prompt(self) -> str:
        """
        构建兼容模式的系统提示词
        包含所有可用工具的详细描述和调用格式
        
        Returns:
            系统提示词字符串
        """
        # 加载提示词模板
        try:
            template = prompt_loader.load_prompt("utils", "tool_compatibility")
        except FileNotFoundError:
            if self.logger:
                self.logger.warning("未找到工具兼容模式提示词模板，使用默认模板")
            raise FileNotFoundError("未找到工具兼容模式提示词模板")
            # 如果模板文件不存在，使用默认模板（向后兼容）  
        # 构建工具描述
        tools_desc = []
        for tool_name, tool_info in self._tools.items():
            schema = tool_info["schema"]
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # 构建参数列表
            params_list = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                params_list.append(
                    f"  - {param_name} ({param_type}){' [必需]' if is_required else ' [可选]'}: {param_desc}"
                )
            
            tool_desc = f"""\n ### {tool_name} \n描述: {tool_info['description']} \n参数:{chr(10).join(params_list) if params_list else '  无参数'} """
            tools_desc.append(tool_desc)
        
        # 替换模板中的占位符
        system_prompt = template.replace("{TOOLS_DESCRIPTION}", ''.join(tools_desc))
        
        return system_prompt
    
    def _extract_tool_calls_from_text(self, text: str) -> Optional[List[Dict]]:
        """
        从模型的文本输出中提取工具调用信息
        
        Args:
            text: 模型输出的文本
            
        Returns:
            工具调用列表，如果没有找到则返回 None
        """
        if not text:
            return None
        
        # 尝试提取 JSON 代码块
        import re
        json_pattern = r'```json\s*({[^`]*})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if not matches:
            # 尝试直接解析整个文本作为 JSON
            try:
                data = json.loads(text.strip())
                if "tool_calls" in data:
                    return data["tool_calls"]
            except json.JSONDecodeError:
                pass
            return None
        
        # 解析第一个 JSON 块
        try:
            data = json.loads(matches[0])
            if "tool_calls" in data and isinstance(data["tool_calls"], list):
                if self.logger:
                    self.logger.info(f"✅ 从文本中提取到 {len(data['tool_calls'])} 个工具调用")
                    for i, tc in enumerate(data["tool_calls"]):
                        self.logger.debug(f"  工具 {i+1}: {tc.get('name', 'unknown')}")
                return data["tool_calls"]
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(f"⚠️  JSON 解析失败: {e}")
                self.logger.debug(f"原始文本: {matches[0]}")
        
        return None
    
    def _execute_tool(self, tool_name: str, arguments: Union[str, Dict]) -> Dict:
        """
        执行已注册的工具函数
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数（JSON字符串或字典）
            
        Returns:
            包含工具执行详细信息的字典：
            - result: 工具执行结果（字符串）
            - success: 是否成功执行
            - elapsed: 执行耗时（秒）
            - tool_name: 工具名称
            - arguments: 工具参数
        """
        if tool_name not in self._tools:
            return {
                "result": f"错误：未找到工具 '{tool_name}'",
                "success": False,
                "elapsed": 0,
                "tool_name": tool_name,
                "arguments": arguments
            }
        
        try:
            # 解析参数
            if isinstance(arguments, str):
                args_dict = json.loads(arguments)
            else:
                args_dict = arguments
            
            # 执行工具函数
            tool_func = self._tools[tool_name]["func"]
            
            if self.logger:
                self.logger.debug(f"执行工具: {tool_name}, 参数: {args_dict}")
            
            start_time = time.time()
            
            # 尝试传递logger参数（如果工具支持）
            import inspect
            sig = inspect.signature(tool_func)
            if 'logger' in sig.parameters:
                result = tool_func(**args_dict, logger=self.logger)
            else:
                result = tool_func(**args_dict)
            
            elapsed = time.time() - start_time
            
            if self.logger:
                self.logger.debug(f"工具执行完成: {tool_name}, 耗时: {elapsed:.3f}s")
            
            # 确保返回字符串
            if not isinstance(result, str):
                result = str(result)
                
            return {
                "result": result,
                "success": True,
                "elapsed": elapsed,
                "tool_name": tool_name,
                "arguments": args_dict
            }
            
        except json.JSONDecodeError as e:
            error_msg = f"参数解析失败: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return {
                "result": error_msg,
                "success": False,
                "elapsed": 0,
                "tool_name": tool_name,
                "arguments": arguments
            }
        except Exception as e:
            error_msg = f"工具执行失败: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            return {
                "result": error_msg,
                "success": False,
                "elapsed": 0,
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    def _call_with_tools_compatibility(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float,
        max_tokens: int,
        max_tool_iterations: int
    ) -> Dict:
        """
        兼容模式：使用提示词工程模拟工具调用
        适用于不支持原生工具调用的模型
        
        Returns:
            包含 content, messages, tool_calls, total_tokens 的字典
        """
        if self.logger:
            self.logger.info("🔧 使用兼容模式进行工具调用（提示词工程）")
        
        # 构建包含工具信息的系统提示词
        compatibility_prompt = self._build_compatibility_system_prompt()
        
        # 复制消息列表并添加系统提示词
        conversation = messages.copy()
        
        # 检查是否已有系统提示词
        has_system = any(msg.get("role") == "system" for msg in conversation)
        if has_system:
            # 合并系统提示词
            for msg in conversation:
                if msg.get("role") == "system":
                    msg["content"] = compatibility_prompt + "\n\n" + msg["content"]
                    break
        else:
            # 在开头插入系统提示词
            conversation.insert(0, {"role": "system", "content": compatibility_prompt})
        
        # 工具调用记录
        tool_call_records = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # 迭代处理工具调用
        for iteration in range(max_tool_iterations):
            if self.logger:
                self.logger.info(f"=" * 80)
                self.logger.info(f"兼容模式迭代 {iteration + 1}/{max_tool_iterations}")
                self.logger.info(f"=" * 80)
                
                # 检查上下文长度，避免payload过大
                total_chars = sum(len(str(m.get('content', ''))) for m in conversation)
                if total_chars > 50000:  # 约50K字符
                    self.logger.warning(f"⚠️ 上下文长度较大: {total_chars} 字符，可能导致响应缓慢")
            
            # 调用 LLM（不使用工具API）- 显式设置timeout避免长时间等待
            try:
                if isinstance(self.client, ZhipuAiClient):
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        thinking={"type": "disabled"},
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=120  # 显式设置2分钟超时
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=120  # 显式设置2分钟超时
                    )
                
                # 记录 token 使用
                record_token_usage_from_resp(response)
                if hasattr(response, 'usage'):
                    usage = response.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                
            except Exception as e:
                # 特殊处理timeout错误（兼容模式）
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    if self.logger:
                        self.logger.error(
                            f"❌ LLM调用超时 (兼容模式迭代 {iteration + 1}/{max_tool_iterations}): {e}\n"
                            f"   上下文消息数: {len(conversation)}\n"
                            f"   建议: 1) 检查网络连接 2) 减少工具调用次数 3) 简化prompt",
                            exc_info=True
                        )
                    # 如果是第一次迭代就超时，直接抛出；否则返回已有结果
                    if iteration == 0:
                        raise
                    else:
                        # 返回部分结果，避免完全失败
                        if self.logger:
                            self.logger.warning(f"⚠️ 超时发生在迭代{iteration + 1}，返回已执行的工具结果")
                        return {
                            "success": False,
                            "error": f"第{iteration + 1}次迭代超时",
                            "content": f"工具调用在第{iteration + 1}次迭代时超时，已执行{iteration}次迭代",
                            "tool_calls": tool_call_records,
                            "iterations": iteration,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens
                        }
                else:
                    if self.logger:
                        self.logger.error(f"LLM调用失败: {e}", exc_info=True)
                    raise
            
            # 获取响应
            message = response.choices[0].message
            content = extract_message_content(message)
            
            if self.logger:
                self.logger.debug(f"响应全部内容长度: {len(content)}")
            
            # 尝试从文本中提取工具调用
            tool_calls = self._extract_tool_calls_from_text(content)
            
            if not tool_calls:
                # 没有工具调用，说明模型完成了回复
                if self.logger:
                    self.logger.info("=" * 80)
                    self.logger.info(f"🎯 模型完成回复（兼容模式，第 {iteration} 次迭代）")
                    self.logger.info(f"   总工具调用次数: {len(tool_call_records)}")
                    self.logger.info(f"   总Token消耗: Prompt={total_prompt_tokens}, Completion={total_completion_tokens}")
                    self.logger.info("=" * 80)
                
                # 添加助手消息到对话
                conversation.append({"role": "assistant", "content": content})
                
                return {
                    "content": content,
                    "messages": conversation,
                    "tool_calls": tool_call_records,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens
                }
            
            # 有工具调用，执行工具
            if self.logger:
                self.logger.info(f"✅ 检测到 {len(tool_calls)} 个工具调用")
            
            # 添加助手消息（包含工具调用请求）
            conversation.append({"role": "assistant", "content": content})
            
            # 执行每个工具
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                if not tool_name:
                    if self.logger:
                        self.logger.warning(f"⚠️  工具调用缺少名称: {tool_call}")
                    continue
                
                if self.logger:
                    self.logger.info(f"执行工具: {tool_name}")
                    self.logger.debug(f"工具参数: {json.dumps(tool_args, ensure_ascii=False)}")
                
                # 执行工具
                tool_exec_info = self._execute_tool(tool_name, tool_args)
                tool_result = tool_exec_info["result"]
                
                if self.logger:
                    if tool_exec_info["success"]:
                        self.logger.info(f"✅ 工具执行成功，耗时: {tool_exec_info['elapsed']:.2f}秒")
                        self.logger.debug(f"完整结果长度: {len(tool_result)}")
                    else:
                        self.logger.error(f"❌ 工具执行失败: {tool_result}")
                
                # 记录工具调用
                tool_call_records.append({
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                    "result": tool_result,
                    "result_preview": tool_result[:200] + "..." if len(tool_result) > 200 else tool_result,
                    "success": tool_exec_info["success"],
                    "elapsed": tool_exec_info["elapsed"]
                })
                
                # 准备工具结果文本（截断过长结果）
                tool_results.append(f"工具 {tool_name} 执行结果:\n{tool_result[:5000]}")
            
            # 将工具结果作为用户消息添加到对话
            if tool_results:
                results_text = "\n\n".join(tool_results)
                conversation.append({
                    "role": "user",
                    "content": f"以下是工具执行结果，请基于这些结果回答用户的问题：\n\n{results_text}"
                })
        
        # 达到最大迭代次数
        if self.logger:
            self.logger.warning("=" * 80)
            self.logger.warning(f"⚠️  达到最大工具调用迭代次数: {max_tool_iterations}")
            self.logger.warning(f"   总工具调用次数: {len(tool_call_records)}")
            self.logger.warning("=" * 80)
        
        return {
            "content": "达到最大工具调用次数限制",
            "messages": conversation,
            "tool_calls": tool_call_records,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        }
    
    def call_with_tools(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_tool_iterations: int = 5,
        tool_choice: Union[str, Dict] = "auto"
    ) -> Dict:
        """
        调用 LLM 并支持工具调用
        
        Args:
            messages: 消息列表
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            max_tool_iterations: 最大工具调用迭代次数
            tool_choice: 工具选择策略（"auto", "none", "required" 或 {"type": "function", "function": {"name": "tool_name"}}）
            
        Returns:
            包含以下字段的字典：
            - content: 最终回复内容
            - messages: 完整的消息历史（包括工具调用）
            - tool_calls: 工具调用记录列表
            - total_tokens: 总token消耗
        """
        if not self._tools:
            # 没有注册工具，直接调用
            if self.logger:
                self.logger.warning("未注册任何工具，将进行普通调用")
            return self.call_without_tools(messages, model_name, temperature, max_tokens)
        
        # 如果强制使用兼容模式，直接调用兼容实现
        if self.force_compatibility_mode or self._tool_call_mode == 'compatibility':
            return self._call_with_tools_compatibility(
                messages, model_name, temperature, max_tokens, max_tool_iterations
            )
        
        # 构建工具定义
        tool_definitions = self._build_tool_definitions()
        
        # 复制消息列表（避免修改原始列表）
        conversation = messages.copy()
        
        # 工具调用记录
        tool_call_records = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # 迭代处理工具调用
        for iteration in range(max_tool_iterations):
            if self.logger:
                self.logger.info(f"=" * 80)
                self.logger.info(f"工具调用迭代 {iteration + 1}/{max_tool_iterations}")
                self.logger.info(f"=" * 80)
                
                # 检查上下文长度
                total_chars = sum(len(str(m.get('content', ''))) for m in conversation)
                if total_chars > 50000:
                    self.logger.warning(f"⚠️ 上下文长度较大: {total_chars} 字符，可能导致响应缓慢")
            
            # 准备请求参数（显式添加timeout）
            request_kwargs = {
                "model": model_name,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 120  # 显式设置2分钟超时
            }
            
            # 添加工具定义
            if self.use_tools_api:
                request_kwargs["tools"] = tool_definitions
                request_kwargs["tool_choice"] = tool_choice
                if self.logger:
                    self.logger.debug(f"使用新版 tools API")
                    self.logger.debug(f"工具数量: {len(tool_definitions)}")
                    self.logger.debug(f"tool_choice: {tool_choice}")
            else:
                request_kwargs["functions"] = tool_definitions
                request_kwargs["function_call"] = tool_choice
                if self.logger:
                    self.logger.debug(f"使用旧版 functions API")
                    self.logger.debug(f"函数数量: {len(tool_definitions)}")
            
            # 记录请求详情（用于调试）
            if self.logger:
                self.logger.debug(f"请求参数:")
                self.logger.debug(f"  - 模型: {model_name}")
                self.logger.debug(f"  - 温度: {temperature}")
                self.logger.debug(f"  - 最大tokens: {max_tokens}")
                self.logger.debug(f"  - 消息数量: {len(conversation)}")
                self.logger.debug(f"完整请求 (不含工具定义):")
                request_log = {k: v for k, v in request_kwargs.items() if k not in ['tools', 'functions']}
                self.logger.debug(json.dumps(request_log, indent=2, ensure_ascii=False))
            
            # 调用 LLM
            try:
                if isinstance(self.client, ZhipuAiClient):
                    # 智谱 AI 特殊处理
                    if self.logger:
                        self.logger.info("使用 ZhipuAI 客户端")
                    request_kwargs.pop("tool_choice", None)  # 智谱可能不支持
                    request_kwargs.pop("function_call", None)
                    response = self.client.chat.completions.create(**request_kwargs)
                else:
                    if self.logger:
                        self.logger.info(f"使用 OpenAI 兼容客户端: {type(self.client).__name__}")
                    response = self.client.chat.completions.create(**request_kwargs)
                
                # 记录 token 使用
                record_token_usage_from_resp(response)
                if hasattr(response, 'usage'):
                    usage = response.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                
            except Exception as e:
                # 特殊处理timeout错误（原生模式）
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    if self.logger:
                        self.logger.error(
                            f"❌ LLM调用超时 (原生模式迭代 {iteration + 1}/{max_tool_iterations}): {e}\n"
                            f"   上下文消息数: {len(conversation)}\n"
                            f"   建议: 1) 检查网络连接 2) 减少max_tool_iterations 3) 简化prompt",
                            exc_info=True
                        )
                    # 如果是第一次迭代就超时，直接抛出；否则返回已有结果
                    if iteration == 0:
                        raise
                    else:
                        # 返回部分结果
                        if self.logger:
                            self.logger.warning(f"⚠️ 超时发生在迭代{iteration + 1}，返回已执行的工具结果")
                        # 构造返回格式（原生模式）
                        return {
                            "success": False,
                            "error": f"第{iteration + 1}次迭代超时",
                            "content": f"工具调用在第{iteration + 1}次迭代时超时，已执行{len(tool_call_records)}个工具",
                            "tool_calls": tool_call_records,
                            "messages": conversation,
                            "iterations": iteration,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens
                        }
                else:
                    if self.logger:
                        self.logger.error(f"LLM调用失败: {e}", exc_info=True)
                    raise
            
            # 获取响应消息
            message = response.choices[0].message
            
            if self.logger:
                if hasattr(message, 'function_call'):
                    self.logger.debug(f"function_call 属性值: {message.function_call}")
            
            # 检查是否有工具调用
            has_tool_calls = False
            
            if self.use_tools_api:
                # 新版 API：检查 tool_calls
                tool_calls = getattr(message, 'tool_calls', None)
                
                if self.logger:
                    if tool_calls:
                        self.logger.info(f"✅ 检测到 {len(tool_calls)} 个工具调用")
                        for i, tc in enumerate(tool_calls):
                            self.logger.info(f"  工具调用 {i+1}: {tc.function.name}")
                    else:
                        self.logger.warning(f"⚠️  未检测到工具调用 (tool_calls = {tool_calls})")
                        self.logger.warning(f"   这可能意味着:")
                        self.logger.warning(f"   1. 模型认为不需要调用工具")
                        self.logger.warning(f"   2. 模型不支持工具调用")
                        self.logger.warning(f"   3. 工具定义格式不正确")
                
                if tool_calls:
                    has_tool_calls = True
                    
                    # 添加助手消息到对话
                    conversation.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in tool_calls
                        ]
                    })
                    
                    # 执行每个工具调用
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments
                        tool_call_id = tool_call.id
                        
                        if self.logger:
                            self.logger.info(f"执行工具: {tool_name}")
                            self.logger.debug(f"工具参数: {tool_args}")
                        
                        # 执行工具
                        tool_exec_info = self._execute_tool(tool_name, tool_args)
                        tool_result = tool_exec_info["result"]
                        
                        if self.logger:
                            if tool_exec_info["success"]:
                                self.logger.info(f"✅ 工具执行成功，耗时: {tool_exec_info['elapsed']:.2f}秒")
                                self.logger.debug(f"结果预览: {tool_result}...")
                            else:
                                self.logger.error(f"❌ 工具执行失败: {tool_result}")
                        
                        # 记录工具调用（包含完整输出）
                        tool_call_records.append({
                            "name": tool_name,
                            "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args, ensure_ascii=False),
                            "result": tool_result,  # 保存完整结果
                            "result_preview": tool_result if len(tool_result) > 200 else tool_result,  # 预览版本
                            "success": tool_exec_info["success"],
                            "elapsed": tool_exec_info["elapsed"]
                        })
                        
                        # 添加工具结果到对话
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_result
                        })
            else:
                # 旧版 API：检查 function_call
                function_call = getattr(message, 'function_call', None)
                if function_call:
                    has_tool_calls = True
                    
                    # 添加助手消息到对话
                    conversation.append({
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": function_call.name,
                            "arguments": function_call.arguments
                        }
                    })
                    
                    # 执行工具
                    tool_name = function_call.name
                    tool_args = function_call.arguments
                    tool_exec_info = self._execute_tool(tool_name, tool_args)
                    tool_result = tool_exec_info["result"]
                    
                    # 记录工具调用（包含完整输出）
                    tool_call_records.append({
                        "name": tool_name,
                        "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args, ensure_ascii=False),
                        "result": tool_result,  # 保存完整结果
                        "result_preview": tool_result[:200] + "..." if len(tool_result) > 200 else tool_result,
                        "success": tool_exec_info["success"],
                        "elapsed": tool_exec_info["elapsed"]
                    })
                    
                    # 添加工具结果到对话
                    conversation.append({
                        "role": "function",
                        "name": tool_name,
                        "content": tool_result
                    })
            
            # 如果没有工具调用，说明模型已完成回复
            if not has_tool_calls:
                final_content = extract_message_content(message)
                
                # 检查是否需要回退到兼容模式
                # 如果是第一次迭代且启用了兼容模式且 tool_choice 要求调用工具，则尝试兼容模式
                if (iteration == 0 and 
                    self.enable_compatibility_mode and 
                    not self.force_compatibility_mode and
                    tool_choice in ["required", "any"] and
                    len(tool_call_records) == 0):
                    
                    if self.logger:
                        self.logger.warning("⚠️  原生工具调用未触发，但 tool_choice 要求调用工具")
                        self.logger.info("🔄 自动回退到兼容模式（提示词工程）")
                    
                    # 切换到兼容模式并重新调用
                    self._tool_call_mode = 'compatibility'
                    return self._call_with_tools_compatibility(
                        messages, model_name, temperature, max_tokens, max_tool_iterations
                    )
                
                if self.logger:
                    self.logger.info("=" * 80)
                    self.logger.info(f"🎯 模型完成回复（第 {iteration + 1} 次迭代）")
                    self.logger.info(f"   总工具调用次数: {len(tool_call_records)}")
                    self.logger.info(f"   总Token消耗: Prompt={total_prompt_tokens}, Completion={total_completion_tokens}")
                    self.logger.info("=" * 80)
                
                return {
                    "content": final_content,
                    "messages": conversation,
                    "tool_calls": tool_call_records,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens
                }
        
        # 达到最大迭代次数
        if self.logger:
            self.logger.warning("=" * 80)
            self.logger.warning(f"⚠️  达到最大工具调用迭代次数: {max_tool_iterations}")
            self.logger.warning(f"   总工具调用次数: {len(tool_call_records)}")
            self.logger.warning("=" * 80)
        if self.logger:
            self.logger.warning(f"达到最大工具调用迭代次数: {max_tool_iterations}")
        
        return {
            "content": "达到最大工具调用次数限制",
            "messages": conversation,
            "tool_calls": tool_call_records,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        }
    
    def call_without_tools(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float,
        max_tokens: int
    ) -> Dict:
        """
        不使用工具的普通调用
        
        Returns:
            包含 content, messages, tool_calls, total_tokens 的字典
        """
        try:
            if isinstance(self.client, ZhipuAiClient):
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    thinking={"type": "disabled"},
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            self.logger.info(f"LLM响应接收成功 \n {response}")
            # 记录 token
            record_token_usage_from_resp(response)
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage'):
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
            
            content = extract_message_content(response.choices[0].message)
            
            return {
                "content": content,
                "messages": messages + [{"role": "assistant", "content": content}],
                "tool_calls": [],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM调用失败: {e}", exc_info=True)
            raise
    
    def get_registered_tools(self) -> List[str]:
        """获取已注册的工具名称列表"""
        return list(self._tools.keys())
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        注销工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否成功注销
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            if self.logger:
                self.logger.debug(f"注销工具: {tool_name}")
            return True
        return False


# 文本嵌入客户端
class EmbeddingClient:
    """文本嵌入客户端，用于计算文本向量"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_EMBEDDING_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_EMBEDDING_MODEL_URL", "https://a1.aizex.me/v1")).rstrip("/")
        self.model = model
        self.total_tokens = 0  # 累计token消耗
        
        if not self.api_key:
            raise ValueError("Embedding API key not provided! Please set OPENAI_EMBEDDING_API_KEY environment variable.")
    
    def get_embeddings(self, texts: List[str], logger=None, max_tokens_per_batch: int = 6000) -> List[np.ndarray]:
        """
        批量获取文本嵌入向量（按token数智能分批，避免API限制）
        
        Args:
            texts: 文本列表
            logger: 日志记录器（可选）
            max_tokens_per_batch: 每批最大token数（默认6000，保守估计避免超过8192限制）
        
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        # 过滤空文本
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 估算token数（更保守的估计：中文约2token/字符，英文约0.4token/字符）
        def estimate_tokens(text: str) -> int:
            """保守估算文本的token数（向上估算避免超限）"""
            # 统计中英文字符
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            # 中文按2倍（保守），英文按0.4倍，再加20%缓冲
            estimated = int((chinese_chars * 2.0 + other_chars * 0.4) * 1.2)
            return max(estimated, 10)  # 至少10 tokens
        
        # 智能分批：按token数分批
        all_embeddings = []
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for text in texts:
            text_tokens = estimate_tokens(text)
            
            # 如果单个文本就超过限制，单独处理（并警告）
            if text_tokens > max_tokens_per_batch:
                if logger:
                    logger.warning(f"单个文本估算{text_tokens} tokens超过限制{max_tokens_per_batch}，将单独处理")
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                batches.append([text])  # 单独一批
                continue
            
            # 如果加入当前批次会超过限制，先保存当前批次
            if current_batch_tokens + text_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                current_batch.append(text)
                current_batch_tokens += text_tokens
        
        # 保存最后一批
        if current_batch:
            batches.append(current_batch)
        
        total_batches = len(batches)
        
        # 处理每一批
        for batch_num, batch_texts in enumerate(batches, 1):
            data = {
                "model": self.model,
                "input": batch_texts
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                
                # 统计token使用
                usage = result.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)
                self.total_tokens += tokens_used
                
                # 提取向量并转换为numpy数组
                batch_embeddings = [np.array(item["embedding"]) for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                
                # 日志输出
                if total_batches > 1:
                    log_msg = f"  [Embedding] 批次{batch_num}/{total_batches}: 文本数={len(batch_texts)}, Token={tokens_used}, 累计={self.total_tokens}"
                else:
                    log_msg = f"  [Embedding] 文本数: {len(batch_texts)}, Token消耗: {tokens_used}, 累计: {self.total_tokens}"
                
                if logger:
                    logger.debug(log_msg)
                else:
                    print(log_msg)
                    
            except requests.exceptions.HTTPError as e:
                # 处理HTTP错误，提供更详细的错误信息
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(error_data))
                    error_detail = f" - {error_msg}"
                    
                    # 如果是token超限错误，尝试处理
                    if "maximum context length" in error_msg.lower():
                        if len(batch_texts) > 1:
                            # 多个文本：拆分成两半递归处理
                            if logger:
                                logger.warning(f"批次{batch_num}超限({len(batch_texts)}个文本)，尝试拆分重试...")
                            mid = len(batch_texts) // 2
                            first_half = self.get_embeddings(batch_texts[:mid], logger=logger, max_tokens_per_batch=max_tokens_per_batch)
                            second_half = self.get_embeddings(batch_texts[mid:], logger=logger, max_tokens_per_batch=max_tokens_per_batch)
                            all_embeddings.extend(first_half)
                            all_embeddings.extend(second_half)
                            continue  # 跳过本次错误，使用递归结果
                        else:
                            # 单个文本就超限：截断文本重试
                            text = batch_texts[0]
                            if logger:
                                logger.warning(f"单个文本超限(原长度{len(text)}字符)，截断到50%重试...")
                            # 截断到一半长度
                            truncated = text[:len(text)//2]
                            retry_embeddings = self.get_embeddings([truncated], logger=logger, max_tokens_per_batch=max_tokens_per_batch)
                            all_embeddings.extend(retry_embeddings)
                            continue  # 跳过本次错误，使用截断结果
                except:
                    error_detail = f" - {e.response.text[:200]}"
                
                raise RuntimeError(f"获取文本嵌入失败(批次{batch_num}/{total_batches}): {e}{error_detail}")
            except Exception as e:
                raise RuntimeError(f"获取文本嵌入失败(批次{batch_num}/{total_batches}): {e}")
        
        return all_embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
        
        Returns:
            相似度值 (0-1)
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



# ========== 便利函数 ==========

def create_tool_client(
    config: Optional[Any] = None,
    tools: Optional[List[Callable]] = None,
    use_tools_api: bool = True,
    logger = None
) -> LLMToolClient:
    """
    创建 LLM 工具客户端的便利函数
    
    Args:
        config: 模型配置对象
        tools: 预注册的工具函数列表
        use_tools_api: True使用新版tools API，False使用旧版functions API
        logger: 日志记录器
        
    Returns:
        LLMToolClient 实例
    
    示例:
        @tool
        def my_tool(arg: str) -> str:
            return "result"
        
        client = create_tool_client(config, tools=[my_tool])
    """
    return LLMToolClient(
        config=config,
        tools=tools,
        use_tools_api=use_tools_api,
        logger=logger
    )


def get_tool_definitions(tools: List[Callable], use_tools_api: bool = True) -> List[Dict]:
    """
    获取工具定义列表（不创建客户端）
    
    适用于需要手动处理工具调用的场景
    
    Args:
        tools: 工具函数列表
        use_tools_api: True使用新版tools API格式，False使用旧版functions格式
        
    Returns:
        工具定义列表
    
    示例:
        @tool
        def search(query: str) -> str:
            return "result"
        
        definitions = get_tool_definitions([search])
        # 手动构建请求
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[...],
            tools=definitions
        )
    """
    # 创建临时客户端来生成定义（使用patch避免创建真实客户端）
    from unittest.mock import Mock, patch
    
    with patch('src.utils.llm_client.create_llm_client') as mock_create:
        mock_create.return_value = Mock()
        
        mock_config = Mock()
        mock_config.api_key = "dummy"
        mock_config.model_url = "dummy"
        mock_config.timeout = 60
        
        temp_client = LLMToolClient(
            config=mock_config,
            tools=tools,
            use_tools_api=use_tools_api
        )
        
        return temp_client._build_tool_definitions()