from src.utils.llm_client import tool

@tool
def hello_world(name: str) -> str:
    '''打印问候语 (INFP专属工具)
    说明：这里是一个简单的示例工具，接受一个名字参数并返回问候语。
    注意：需要把工具尽可能设计详细，包括参数说明和返回值说明。
    Args:
        name: 需要问候的名字
    Returns:
        问候语字符串
    
    '''
    return f"Hello, {name}! I am INFP."

if __name__ == "__main__":
    # 测试工具函数
    print(hello_world("INFP"))
