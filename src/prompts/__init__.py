"""
Prompt 管理模块
集中管理所有 prompt 文件，支持性格与功能的组合
"""
from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """Prompt 加载器，支持模块化的 prompt 组合"""
    
    def __init__(self):
        self.prompt_dir = Path(__file__).parent
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, category: str, name: str, use_cache: bool = True) -> str:
        """
        加载 prompt 文件（兼容旧接口）
        
        Args:
            category: prompt 类别 (如 'mbti')
            name: prompt 名称 (如 'intj_prompt')
            use_cache: 是否使用缓存
            
        Returns:
            prompt 内容字符串
        """
        cache_key = f"{category}/{name}"
        
        # 如果使用缓存且缓存中存在,直接返回
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 构建文件路径
        prompt_file = self.prompt_dir / category / f"{name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_file}")
        
        # 读取文件内容
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 存入缓存
        self._cache[cache_key] = content
        
        return content
    
    def load_personality(self, mbti_type: str, use_cache: bool = True) -> str:
        """
        加载性格核心提示词
        
        Args:
            mbti_type: MBTI 类型 (如 'intj', 'infj')
            use_cache: 是否使用缓存
            
        Returns:
            性格核心提示词内容
        """
        cache_key = f"mbti/{mbti_type.lower()}/personality"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 新的目录结构: mbti/{type}/personality.txt
        prompt_file = self.prompt_dir / "mbti" / mbti_type.lower() / "personality.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Personality prompt 不存在: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._cache[cache_key] = content
        return content
    
    def load_function(self, mbti_type: str, function_name: str, use_cache: bool = True) -> str:
        """
        加载功能提示词
        
        Args:
            mbti_type: MBTI 类型 (如 'intj', 'infj')
            function_name: 功能名称 (如 'evaluate_solution', 'analyze_task')
            use_cache: 是否使用缓存
            
        Returns:
            功能提示词内容
        """
        cache_key = f"mbti/{mbti_type.lower()}/{function_name}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 新的目录结构: mbti/{type}/{function}.txt
        prompt_file = self.prompt_dir / "mbti" / mbti_type.lower() / f"{function_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Function prompt 不存在: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._cache[cache_key] = content
        return content
    
    def combine_prompts(
        self,
        mbti_type: str,
        function_name: str,
        shared_template: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        组合性格提示词和功能提示词
        
        Args:
            mbti_type: MBTI 类型
            function_name: 功能名称
            shared_template: 可选的共享模板名称
            use_cache: 是否使用缓存
            
        Returns:
            组合后的完整 prompt
        """
        # 加载性格核心
        personality = self.load_personality(mbti_type, use_cache)
        
        # 加载功能提示词
        function_prompt = self.load_function(mbti_type, function_name, use_cache)
        
        # 组合: 性格核心 + 空行 + 功能提示词
        combined = f"{personality}\n\n{function_prompt}"
        
        # 如果需要共享模板，追加在最后
        if shared_template:
            shared_file = self.prompt_dir / "mbti" / "shared" / f"{shared_template}.txt"
            if shared_file.exists():
                with open(shared_file, 'r', encoding='utf-8') as f:
                    shared_content = f.read()
                combined = f"{combined}\n\n{shared_content}"
        
        return combined
    
    def load_system_prompt(self, prompt_name: str, use_cache: bool = True) -> str:
        """
        加载系统级提示词
        
        Args:
            prompt_name: 系统提示词名称 (如 'team_formation_judge')
            use_cache: 是否使用缓存
            
        Returns:
            系统提示词内容
        """
        cache_key = f"system/{prompt_name}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 系统提示词路径: system/{prompt_name}.txt
        prompt_file = self.prompt_dir / "system" / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"System prompt 不存在: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        self._cache[cache_key] = content
        return content
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


# 全局 prompt 加载器实例
prompt_loader = PromptLoader()










