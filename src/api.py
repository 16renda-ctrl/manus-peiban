import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langgraph.pregel import StreamMode
from typing import TypedDict, Literal
from datetime import datetime # 修正：添加 datetime 导入

# 导入核心图逻辑
# 假设 core_graph.p# 导入核心图逻辑
import core_graph 

# --- FastAPI Setup ---app = FastAPI(
    title="Companion Robot Cognitive API",
    description="Real-time WebSocket API for streaming LangGraph execution trace.",
)

# 允许跨域访问，方便前端开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，开发阶段方便
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 编译 LangGraph
companion_graph = core_graph.build_companion_graph()
PERSONALITY_MASKS = core_graph.PERSONALITY_MASKS

# --- WebSocket Endpoint ---
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    
    # 初始化会话状态
    current_state = {
        "conversation_history": [],
        "current_personality": "mentor", # 默认人格
    }

    try:
        while True:
            # 1. 接收用户输入和人格选择
            data = await websocket.receive_text()
            message = json.loads(data)
            
            user_input = message.get("user_input", "").strip()
            personality = message.get("personality", "mentor").strip().lower()
            
            if not user_input:
                continue

            # 更新当前人格
            current_state["current_personality"] = personality
            
            # 准备图的输入状态
            input_state = core_graph.CompanionState(
                user_input=user_input,
                current_personality=personality,
                conversation_history=current_state["conversation_history"],
                detected_emotion="",
                should_use_skill=False,
                skill_to_use="",
                skill_result="",
                final_response="",
            )

            # 2. 发送开始信号
            await websocket.send_json({
                "type": "start",
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "personality": PERSONALITY_MASKS.get(personality, {}).get("name", personality),
            })

            # 3. 实时流式传输 LangGraph 执行轨迹
            final_response_parts = []
            full_response_buffer = ""
            
            # 使用 astream 实时获取每个节点的输出
            async for step in companion_graph.astream(input_state, stream_mode=StreamMode.updates):
                # step 是一个字典，键是节点名，值是该节点对状态的更新
                node_name = list(step.keys())[0]
                state_update = step[node_name]
                
                # 提取关键信息进行可视化
                trace_data = {
                    "type": "trace",
                    "node": node_name,
                    "update": state_update,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # 特殊处理 final_response 的流式输出
                if node_name == "generate_response" and "final_response" in state_update:
                    # 这是一个简化的流式处理，实际 LLM 流式输出需要更复杂的集成
                    # 这里我们假设 final_response 是一个完整的字符串，但可以模拟分块发送
                    
                    # 模拟流式输出
                    full_response_buffer = state_update["final_response"]
                    
                    # 发送节点完成信号
                    await websocket.send_json(trace_data)
                    
                else:
                    # 发送节点执行信号
                    await websocket.send_json(trace_data)
                    
            # 4. 模拟流式输出 final_response
            # 由于 LangGraph 的 astream 默认是状态更新，我们无法直接获取 LLM 的 token 流
            # 这里我们发送最终的完整回复，前端可以一次性显示
            await websocket.send_json({
                "type": "stream",
                "content": full_response_buffer,
            })
            
            # 5. 更新历史记录
            current_state["conversation_history"].append({
                "user": user_input,
                "bot": full_response_buffer
            })

            # 6. 发送结束信号
            await websocket.send_json({
                "type": "end",
                "timestamp": datetime.now().isoformat(),
                "final_response": full_response_buffer,
            })

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass # 忽略发送错误时的异常

# --- Uvicorn 启动配置 ---
if __name__ == "__main__":
    import uvicorn
    # 注意：这里需要将工作目录切换到项目根目录，以便正确加载 .env
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    uvicorn.run(app, host="0.0.0.0", port=8000)
