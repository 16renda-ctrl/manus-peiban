"""
伴伴机器人：LangGraph 核心骨架 (Skeleton Implementation)

这是一个最小化的、使用 Mock 数据的骨架实现，用于验证整体流程的逻辑。
真实的 LLM 调用和技能实现将在后续阶段添加。

架构说明：
- State：机器人的"短期记忆"，包含当前对话、选中的人格等。
- Nodes：骨架中的各个处理步骤（决策、技能执行、回复生成）。
- Edges：节点之间的连接逻辑。
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import json
from datetime import datetime


# ============================================================================
# 1. 定义 State（状态）：机器人的"短期记忆"
# ============================================================================

class CompanionState(TypedDict):
    """
    伴伴机器人的状态对象。
    这就像机器人的"短期记忆"，记录当前对话的所有关键信息。
    """
    # 用户输入
    user_input: str
    
    # 当前选中的人格面具
    current_personality: Literal["mentor", "trickster", "guardian"]
    
    # 对话历史（简化版，只保留最近的几条）
    conversation_history: list[dict]
    
    # 检测到的用户情绪
    detected_emotion: str
    
    # 是否需要调用技能
    should_use_skill: bool
    
    # 调用的技能名称
    skill_to_use: str
    
    # 技能执行结果
    skill_result: str
    
    # 最终回复
    final_response: str


# ============================================================================
# 2. 定义人格面具（Personality Masks）
# ============================================================================

PERSONALITY_MASKS = {
    "mentor": {
        "name": "智慧导师 (The Wise Mentor)",
        "core_motivation": "提供指引，共同成长",
        "traits": ["冷静", "博学", "逻辑清晰", "善于提问"],
        "system_prompt": """你是一个智慧的导师。你的核心动机是帮助用户思考和成长。
你的说话风格是：冷静、逻辑清晰、经常提出启发性的问题。
你避免过度情绪化的表达，而是用结构化的建议来帮助用户。""",
    },
    "trickster": {
        "name": "调皮伙伴 (The Playful Trickster)",
        "core_motivation": "打破沉闷，寻找乐趣",
        "traits": ["幽默", "充满好奇", "偶尔毒舌", "不拘小节"],
        "system_prompt": """你是一个调皮、幽默的伙伴。你的核心动机是让对话充满乐趣和惊喜。
你的说话风格是：轻松活泼、经常开玩笑、甚至会用"毒舌"的方式互怼。
你不怕说出有点"不礼貌"但有趣的话，总是试图打破沉闷的气氛。""",
    },
    "guardian": {
        "name": "温柔守护者 (The Gentle Guardian)",
        "core_motivation": "提供安全感，治愈疲惫",
        "traits": ["共情力强", "包容", "细心", "情绪稳定"],
        "system_prompt": """你是一个温柔、富有同情心的守护者。你的核心动机是为用户提供情感支持和安全感。
你的说话风格是：温暖、倾听为主、经常使用肯定和鼓励的语言。
你能感受到用户的疲惫，并用温柔的方式陪伴他们。""",
    },
}


# ============================================================================
# 3. 定义技能（Skills）- Mock 实现
# ============================================================================

class SkillRegistry:
    """技能注册表。在真实实现中，这将连接到向量数据库、API 等。"""
    
    @staticmethod
    def long_term_memory_store(key: str, value: str) -> str:
        """
        好记性技能：存储信息到长期记忆。
        Mock 实现：直接返回确认消息。
        真实实现：会调用向量数据库。
        """
        return f"✓ 已记住：{key} = {value}"
    
    @staticmethod
    def long_term_memory_retrieve(query: str) -> str:
        """
        好记性技能：从长期记忆中检索信息。
        Mock 实现：返回模拟的记忆。
        """
        mock_memories = {
            "用户名字": "小明",
            "用户爱好": "看书、听音乐",
            "用户工作": "程序员",
        }
        return mock_memories.get(query, f"没有找到关于'{query}'的记忆。")
    
    @staticmethod
    def mood_tracker_analyze(user_input: str) -> str:
        """
        心情气压计技能：分析用户的情绪。
        Mock 实现：根据关键词简单判断。
        真实实现：会使用情感分析模型。
        """
        if any(word in user_input for word in ["开心", "高兴", "太好了", "😊"]):
            return "happy"
        elif any(word in user_input for word in ["难过", "伤心", "累", "😢"]):
            return "sad"
        elif any(word in user_input for word in ["生气", "烦", "😠"]):
            return "angry"
        else:
            return "neutral"
    
    @staticmethod
    def shared_experience_fetch(topic: str) -> str:
        """
        共同经历技能：获取外部信息（如新闻、天气）。
        Mock 实现：返回模拟的信息。
        真实实现：会调用 News API、Weather API 等。
        """
        mock_data = {
            "天气": "今天天气晴朗，气温 15°C，适合出门散步。",
            "新闻": "最新的科技新闻：AI 技术继续突破，多智能体系统成为新热点。",
            "音乐": "推荐歌曲：《晴天》- 周杰伦。这首歌很适合现在的心情。",
        }
        return mock_data.get(topic, f"关于'{topic}'的信息暂时不可用。")


# ============================================================================
# 4. 定义节点（Nodes）
# ============================================================================

def node_receive_input(state: CompanionState) -> CompanionState:
    """
    节点 1：接收用户输入
    这是流程的入口。在这里，我们记录用户的输入。
    """
    print(f"\n[节点 1] 接收用户输入: {state['user_input']}")
    return state


def node_analyze_emotion(state: CompanionState) -> CompanionState:
    """
    节点 2：分析用户情绪
    调用"心情气压计"技能，检测用户的情绪。
    """
    print(f"\n[节点 2] 分析用户情绪...")
    emotion = SkillRegistry.mood_tracker_analyze(state["user_input"])
    state["detected_emotion"] = emotion
    print(f"  → 检测到情绪: {emotion}")
    return state


def node_decide_skill(state: CompanionState) -> CompanionState:
    """
    节点 3：决定是否需要调用技能
    这是一个"决策节点"，根据用户输入和情绪，决定是否需要调用技能。
    
    Mock 逻辑：
    - 如果用户输入中包含"记住"，调用"好记性"的存储功能。
    - 如果用户输入中包含"天气"或"新闻"，调用"共同经历"的获取功能。
    - 否则，不调用技能，直接回复。
    """
    print(f"\n[节点 3] 决定是否调用技能...")
    
    user_input = state["user_input"].lower()
    
    if "记住" in user_input or "记一下" in user_input:
        state["should_use_skill"] = True
        state["skill_to_use"] = "long_term_memory_store"
        print(f"  → 决定调用技能: 好记性 (存储)")
    elif "天气" in user_input or "新闻" in user_input or "音乐" in user_input:
        state["should_use_skill"] = True
        state["skill_to_use"] = "shared_experience_fetch"
        print(f"  → 决定调用技能: 共同经历")
    else:
        state["should_use_skill"] = False
        state["skill_to_use"] = ""
        print(f"  → 不需要调用技能，直接回复")
    
    return state


def node_execute_skill(state: CompanionState) -> CompanionState:
    """
    节点 4：执行技能
    如果决定了要调用技能，就在这里执行。
    """
    if not state["should_use_skill"]:
        state["skill_result"] = ""
        return state
    
    print(f"\n[节点 4] 执行技能: {state['skill_to_use']}...")
    
    skill_name = state["skill_to_use"]
    user_input = state["user_input"]
    
    if skill_name == "long_term_memory_store":
        # 简单的 Mock：假设用户输入格式是 "记住 key: value"
        parts = user_input.split(":")
        if len(parts) == 2:
            key = parts[0].replace("记住", "").replace("记一下", "").strip()
            value = parts[1].strip()
            result = SkillRegistry.long_term_memory_store(key, value)
        else:
            result = "格式错误。请用 '记住 key: value' 的格式。"
    elif skill_name == "shared_experience_fetch":
        # 提取关键词
        if "天气" in user_input:
            topic = "天气"
        elif "新闻" in user_input:
            topic = "新闻"
        elif "音乐" in user_input:
            topic = "音乐"
        else:
            topic = "天气"
        result = SkillRegistry.shared_experience_fetch(topic)
    else:
        result = "未知技能"
    
    state["skill_result"] = result
    print(f"  → 技能结果: {result}")
    return state


def node_generate_response(state: CompanionState) -> CompanionState:
    """
    节点 5：生成回复
    根据用户输入、情绪、人格和技能结果，生成最终回复。
    
    Mock 实现：直接拼接字符串。
    真实实现：会调用 LLM（如 GPT-4）。
    """
    print(f"\n[节点 5] 生成回复...")
    
    personality = PERSONALITY_MASKS[state["current_personality"]]
    emotion = state["detected_emotion"]
    skill_result = state["skill_result"]
    
    # Mock 回复生成逻辑
    response_templates = {
        ("mentor", "happy"): f"很高兴看到你这么开心！{skill_result if skill_result else '让我们一起思考这个问题。'}",
        ("mentor", "sad"): f"我能感受到你的情绪。{skill_result if skill_result else '让我们冷静地分析这个问题。'}",
        ("mentor", "neutral"): f"这是个有趣的问题。{skill_result if skill_result else '让我从逻辑的角度来帮你分析。'}",
        ("trickster", "happy"): f"哈哈，你今天心情不错嘛！{skill_result if skill_result else '那咱们来玩点有趣的。'}",
        ("trickster", "sad"): f"怎么了，被打击了？{skill_result if skill_result else '别难过，我来逗你笑。'}",
        ("trickster", "neutral"): f"又来找我玩了？{skill_result if skill_result else '我有个有趣的想法...'}",
        ("guardian", "happy"): f"你的开心感染了我！{skill_result if skill_result else '让我们一起享受这美好的时刻。'}",
        ("guardian", "sad"): f"我能感受到你的疲惫。{skill_result if skill_result else '让我陪陪你，一切都会好的。'}",
        ("guardian", "neutral"): f"有什么我可以帮你的吗？{skill_result if skill_result else '我在这里陪你。'}",
    }
    
    key = (state["current_personality"], emotion)
    response = response_templates.get(key, f"[{personality['name']}] 我听到你说的了。{skill_result if skill_result else ''}")
    
    state["final_response"] = response
    print(f"  → 生成的回复: {response}")
    return state


def node_update_history(state: CompanionState) -> CompanionState:
    """
    节点 6：更新对话历史
    将这一轮的对话添加到历史记录中。
    """
    print(f"\n[节点 6] 更新对话历史...")
    
    state["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "user": state["user_input"],
        "bot": state["final_response"],
        "personality": state["current_personality"],
        "emotion": state["detected_emotion"],
    })
    
    print(f"  → 对话历史已更新（共 {len(state['conversation_history'])} 条）")
    return state


# ============================================================================
# 5. 构建 LangGraph
# ============================================================================

def build_companion_graph():
    """
    构建伴伴机器人的 LangGraph。
    这就是机器人的"大脑"，定义了它如何思考和回复。
    """
    
    # 创建图
    graph = StateGraph(CompanionState)
    
    # 添加节点
    graph.add_node("receive_input", node_receive_input)
    graph.add_node("analyze_emotion", node_analyze_emotion)
    graph.add_node("decide_skill", node_decide_skill)
    graph.add_node("execute_skill", node_execute_skill)
    graph.add_node("generate_response", node_generate_response)
    graph.add_node("update_history", node_update_history)
    
    # 添加边（连接节点）
    graph.add_edge("receive_input", "analyze_emotion")
    graph.add_edge("analyze_emotion", "decide_skill")
    graph.add_edge("decide_skill", "execute_skill")
    graph.add_edge("execute_skill", "generate_response")
    graph.add_edge("generate_response", "update_history")
    graph.add_edge("update_history", END)
    
    # 设置入口点
    graph.set_entry_point("receive_input")
    
    # 编译图
    compiled_graph = graph.compile()
    
    return compiled_graph


# ============================================================================
# 6. 主函数：测试骨架
# ============================================================================

def run_skeleton_test():
    """
    运行骨架测试。
    这个函数演示了整个流程如何工作。
    """
    
    print("=" * 80)
    print("伴伴机器人 - LangGraph 骨架测试")
    print("=" * 80)
    
    # 构建图
    graph = build_companion_graph()
    
    # 初始化状态
    initial_state: CompanionState = {
        "user_input": "",
        "current_personality": "mentor",
        "conversation_history": [],
        "detected_emotion": "neutral",
        "should_use_skill": False,
        "skill_to_use": "",
        "skill_result": "",
        "final_response": "",
    }
    
    # 测试用例
    test_inputs = [
        ("你好，我是来找你的。", "mentor"),
        ("今天天气怎么样？", "trickster"),
        ("我感到很疲惫。", "guardian"),
        ("记住 我的爱好: 编程和阅读", "mentor"),
    ]
    
    for user_input, personality in test_inputs:
        print(f"\n{'=' * 80}")
        print(f"用户输入: {user_input}")
        print(f"选择人格: {PERSONALITY_MASKS[personality]['name']}")
        print(f"{'=' * 80}")
        
        # 更新状态
        initial_state["user_input"] = user_input
        initial_state["current_personality"] = personality
        
        # 运行图
        result = graph.invoke(initial_state)
        
        # 更新状态为下一轮的初始状态
        initial_state = result
        
        print(f"\n最终回复: {result['final_response']}")
    
    print(f"\n{'=' * 80}")
    print("骨架测试完成！")
    print(f"{'=' * 80}")
    
    # 打印完整的对话历史
    print("\n对话历史摘要:")
    print(json.dumps(result["conversation_history"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_skeleton_test()
