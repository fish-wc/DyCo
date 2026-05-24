时间：2025.11.4
# mbti
    目前每一个mbti性格的智能体都有各自的提示词，包括_calculate_speaking_intention.txt、analyze_task.txt、decide_team_preference.txt、evaluate_solution.txt、generate.txt、personality.txt。
    当前的设计思路来看，personality.txt属于性格提示词，其他的提示词都是功能提示词。现在的代码思路是，由性格提示词拼接功能提示词构成完整的提示词。当前为了保留智能体的自由度，每个智能体都有自己的功能提示词，目前为了简化实现，每个智能体的功能提示词设定为一样的。

# system
    系统提示词部分是对功能提共性的部分进行约束。比如输出格式限定等。