'''
有两种websearch工具可供选择： jina和知谱(zhipu)。默认使用jina实现的websearch工具。
要切换到知谱实现的websearch工具，请注释掉导入jina的行，并取消注释导入知谱的行。
'''

from src.tools.system.websearch.websearch_jina import web_search_tool
# from src.tools.system.websearch.websearch_zhipu import web_search_tool