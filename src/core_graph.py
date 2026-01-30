import os
import json
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from openai import OpenAI, AsyncOpenAI # 导入 AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

class CompanionState(TypedDict):
    user_input: str
    current_personality: Literal["mentor", "trickster", "guardian"]
    conversation_history: list[dict]
    detected_emotion: str
    should_use_skill: bool
    skill_to_use: str
    skill_result: str
    final_response: str

def get_openai_client():
    """返回同步客户端，用于兼容旧的 CLI 模式"""
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url or not base_url.endswith("/v1"):
        base_url = (base_url or "") + "/v1"
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )

def get_async_openai_client():
    """返回异步客户端，用于新的 API 模式"""
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url or not base_url.endswith("/v1"):
        base_url = (base_url or "") + "/v1"
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )

PERSONALITY_MASKS = {
    "mentor": {
        "name": "智慧导师 (The Wise Mentor)",
        "system_prompt": "你是一个智慧的导师。你的核心动机是帮助用户思考和成长。说话风格：冷静、逻辑清晰、启发性提问。",
    },
    "trickster": {
        "name": "调皮伙伴 (The Playful Trickster)",
        "system_prompt": "你是一个调皮、幽默的伙伴。核心动机：寻找乐趣。说话风格：轻松活泼、开玩笑、偶尔毒舌。",
    },
    "guardian": {
        "name": "温柔守护者 (The Gentle Guardian)",
        "system_prompt": "你是一个温柔的守护者。核心动机：提供安全感。说话风格：温暖、倾听、肯定和鼓励。",
    },
}

# 所有节点改为异步函数
async def node_receive_input(state: CompanionState) -> CompanionState:
    print(f"\n[节点 1] 接收用户输入: {state['user_input']}")
    return state

async def node_analyze_emotion(state: CompanionState) -> CompanionState:
    print(f"\n[节点 2] 分析用户情绪...")
    client = get_async_openai_client()
    try:
        res = await client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=[{"role": "system", "content": "只返回一个词：happy, sad, angry, neutral"},
                      {"role": "user", "content": state['user_input']}]
        )
        emotion = res.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"  ⚠️ 失败: {e}")
        emotion = "neutral"
    state["detected_emotion"] = emotion
    print(f"  → 情绪: {emotion}")
    return state

async def node_decide_skill(state: CompanionState) -> CompanionState:
    print(f"\n[节点 3] 决策技能...")
    client = get_async_openai_client()
    try:
        res = await client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=[{"role": "system", "content": "判断是否需要工具：long_term_memory_store, shared_experience_fetch。不需要返回 none。"},
                      {"role": "user", "content": state['user_input']}]
        )
        decision = res.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"  ⚠️ 失败: {e}")
        decision = "none"
    
    state["should_use_skill"] = "none" not in decision
    state["skill_to_use"] = decision if state["should_use_skill"] else ""
    print(f"  → 决策: {state['skill_to_use'] or '直接回复'}")
    return state

async def node_execute_skill(state: CompanionState) -> CompanionState:
    if not state["should_use_skill"]:
        state["skill_result"] = ""
        return state
    print(f"\n[节点 4] 执行技能...")
    # 异步模拟技能执行
    await asyncio.sleep(0.1) 
    state["skill_result"] = "模拟数据：操作成功。"
    return state

async def node_generate_response(state: CompanionState) -> CompanionState:
    print(f"\n[节点 5] 生成回复...")
    client = get_async_openai_client()
    personality = PERSONALITY_MASKS[state["current_personality"]]
    try:
        # 使用流式 API，但只取最终结果，以兼容 LangGraph 的状态更新模式
        res = await client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=[
                {"role": "system", "content": personality["system_prompt"]},
                {"role": "user", "content": f"情绪:{state['detected_emotion']}, 技能结果:{state['skill_result']}\n用户说:{state['user_input']}"}
            ]
        )
        state["final_response"] = res.choices[0].message.content
    except Exception as e:
        print(f"  ⚠️ 失败: {e}")
        state["final_response"] = "抱歉，我现在有点累。"
    print(f"  → 回复: {state['final_response'][:30]}...")
    return state

async def node_update_history(state: CompanionState) -> CompanionState:
    # 异步模拟历史记录更新
    await asyncio.sleep(0.01)
    state["conversation_history"].append({"user": state["user_input"], "bot": state["final_response"]})
    return state

def build_companion_graph():
    graph = StateGraph(CompanionState)
    # 节点函数保持不变，但现在它们是异步的
    graph.add_node("receive_input", node_receive_input)
    graph.add_node("analyze_emotion", node_analyze_emotion)
    graph.add_node("decide_skill", node_decide_skill)
    graph.add_node("execute_skill", node_execute_skill)
    graph.add_node("generate_response", node_generate_response)
    graph.add_node("update_history", node_update_history)
    graph.add_edge("receive_input", "analyze_emotion")
    graph.add_edge("analyze_emotion", "decide_skill")
    graph.add_edge("decide_skill", "execute_skill")
    graph.add_edge("execute_skill", "generate_response")
    graph.add_edge("generate_response", "update_history")
    graph.add_edge("update_history", END)
    graph.set_entry_point("receive_input")
    return graph.compile()

if __name__ == "__main__":
    # CLI 模式需要使用同步调用
    import asyncio
    
    app = build_companion_graph()

    print("=" * 60)
    print("欢迎使用伴伴机器人！")
    print("=" * 60)
    print("\n可用人格：")
    for key, value in PERSONALITY_MASKS.items():
        print(f"  • {key}: {value['name']}")
    print()

    while True:
        personality = input("请选择人格 (mentor/trickster/guardian，或输入 'quit' 退出): ").strip().lower()
        if personality == "quit":
            print("再见！")
            break
        if personality not in PERSONALITY_MASKS:
            print("❌ 无效的人格选择，请重试。")
            continue

        user_input = input("请输入你的消息: ").strip()
        if not user_input:
            print("❌ 消息不能为空。")
            continue

        print("\n" + "=" * 60)
        # 在同步环境中运行异步图
        result = asyncio.run(app.ainvoke({
            "user_input": user_input,
            "current_personality": personality,
            "conversation_history": []
        }))
        print("=" * 60)
        print(f"\n【{PERSONALITY_MASKS[personality]['name']}】的回复：\n")
        print(result['final_response'])
        print("\n" + "=" * 60 + "\n")
