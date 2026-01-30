import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph
from typing import TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel

# 导入核心图逻辑
from . import core_graph 

# --- FastAPI Setup ---
app = FastAPI(
    title="Companion Robot Cognitive API",
    description="Real-time WebSocket API for streaming LangGraph execution trace with dynamic configuration.",
)

# 允许跨域访问，方便前端开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 编译 LangGraph
companion_graph = core_graph.build_companion_graph()
PERSONALITY_MASKS = core_graph.PERSONALITY_MASKS

# --- System Configuration Models ---

class NodeConfig(BaseModel):
    """节点配置"""
    id: str
    label: str
    type: str  # "input", "processor", "output"
    angle: float  # 圆形布局的角度

class ConnectionConfig(BaseModel):
    """连接配置"""
    from_node: str
    to_node: str
    color: str  # 连接线颜色

class EmotionConfig(BaseModel):
    """情绪配置"""
    name: str
    color: str  # 十六进制颜色
    intensity: float  # 0-1 强度

class SystemConfig(BaseModel):
    """系统配置"""
    nodes: list[NodeConfig]
    connections: list[ConnectionConfig]
    emotions: list[EmotionConfig]
    personalities: dict[str, dict]

# --- Panel Configuration Models ---

class PanelConfig(BaseModel):
    """面板配置"""
    id: str
    title: str
    type: str  # "status", "metrics", "logs", "memory", "custom"
    icon: str = "📊"
    refreshInterval: int = 1000  # 毫秒
    description: str = ""

class PanelsConfig(BaseModel):
    """面板集合配置"""
    panels: list[PanelConfig]
    layout: str = "vertical"
    maxWidth: str = "400px"

# --- System Configuration Endpoint ---

def get_system_config() -> SystemConfig:
    """
    生成系统配置，前端可以根据此配置动态生成界面
    这样后端升级时，前端会自动适配
    """
    
    # 节点配置（从 core_graph 中提取）
    nodes = [
        NodeConfig(id="receive_input", label="接收输入", type="input", angle=0),
        NodeConfig(id="analyze_emotion", label="情绪分析", type="processor", angle=60),
        NodeConfig(id="decide_skill", label="技能决策", type="processor", angle=120),
        NodeConfig(id="execute_skill", label="执行技能", type="processor", angle=180),
        NodeConfig(id="generate_response", label="生成回复", type="processor", angle=240),
        NodeConfig(id="update_history", label="更新历史", type="output", angle=300),
    ]
    
    # 连接配置
    connections = [
        ConnectionConfig(from_node="receive_input", to_node="analyze_emotion", color="#00BFFF"),
        ConnectionConfig(from_node="analyze_emotion", to_node="decide_skill", color="#00BFFF"),
        ConnectionConfig(from_node="decide_skill", to_node="execute_skill", color="#00BFFF"),
        ConnectionConfig(from_node="execute_skill", to_node="generate_response", color="#9370DB"),
        ConnectionConfig(from_node="generate_response", to_node="update_history", color="#9370DB"),
    ]
    
    # 情绪配置
    emotions = [
        EmotionConfig(name="happy", color="#FFD700", intensity=1.0),
        EmotionConfig(name="sad", color="#4169E1", intensity=0.8),
        EmotionConfig(name="angry", color="#FF4500", intensity=0.9),
        EmotionConfig(name="neutral", color="#00BFFF", intensity=0.6),
    ]
    
    # 人格配置
    personalities = {
        name: {
            "name": config["name"],
            "system_prompt": config["system_prompt"]
        }
        for name, config in PERSONALITY_MASKS.items()
    }
    
    return SystemConfig(
        nodes=nodes,
        connections=connections,
        emotions=emotions,
        personalities=personalities
    )

@app.get("/api/system-config")
async def system_config():
    """
    返回系统配置
    前端在初始化时调用此端点，获取节点、连接、情绪等配置
    """
    config = get_system_config()
    return config.model_dump()

# --- Panels Configuration Endpoint ---

def get_panels_config() -> PanelsConfig:
    """
    生成面板配置，前端根据此配置动态生成侧面板
    后端可以随时添加新面板，前端会自动显示
    """
    
    panels = [
        PanelConfig(
            id="llm-status",
            title="LLM 连接状态",
            type="status",
            icon="🔌",
            description="显示 LLM 服务的连接状态和响应延迟"
        ),
        PanelConfig(
            id="system-metrics",
            title="系统性能",
            type="metrics",
            icon="📊",
            description="实时 CPU、内存、网络使用情况"
        ),
        PanelConfig(
            id="event-logs",
            title="事件日志",
            type="logs",
            icon="📝",
            description="实时系统事件和错误日志"
        ),
        PanelConfig(
            id="memory-usage",
            title="内存管理",
            type="memory",
            icon="💾",
            description="对话历史和缓存内存使用"
        ),
    ]
    
    return PanelsConfig(
        panels=panels,
        layout="vertical",
        maxWidth="400px"
    )

@app.get("/api/panels-config")
async def panels_config():
    """
    返回面板配置
    前端在初始化时调用此端点，获取要显示的所有面板
    """
    config = get_panels_config()
    return config.model_dump()

# --- Panel Data Endpoint (for real-time updates) ---

@app.get("/api/panels-data")
async def panels_data():
    """
    返回所有面板的实时数据
    前端可以定期调用此端点获取最新数据
    """
    import psutil
    
    # LLM 状态
    llm_status = {
        "status": "connected",
        "latency": 45,
    }
    
    # 系统指标
    system_metrics = {
        "cpu": psutil.cpu_percent(interval=0.1),
        "memory": psutil.virtual_memory().percent,
        "network": 0,  # 可以扩展为实际网络使用
    }
    
    # 事件日志（示例）
    event_logs = {
        "logs": [
            {"level": "info", "message": "系统启动完成", "timestamp": "14:30:15"},
            {"level": "info", "message": "WebSocket 连接已建立", "timestamp": "14:30:16"},
        ]
    }
    
    # 内存使用
    memory_usage = {
        "conversationSize": 1024 * 50,  # 50KB
        "cacheSize": 1024 * 100,  # 100KB
    }
    
    return {
        "llm-status": llm_status,
        "system-metrics": system_metrics,
        "event-logs": event_logs,
        "memory-usage": memory_usage,
    }

# --- WebSocket Endpoint ---

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    # 初始化会话状态
    current_state = {
        "conversation_history": [],
        "current_personality": "mentor",
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

            # 2. 发送开始信号（自描述事件）
            await websocket.send_json({
                "type": "start",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "input": user_input,
                    "personality": personality,
                }
            })

            # 3. 实时流式传输 LangGraph 执行轨迹
            full_response_buffer = ""
            
            # 使用 astream 实时获取每个节点的输出
            async for step in companion_graph.astream(input_state):
                node_name = list(step.keys())[0]
                state_update = step[node_name]
                
                # 发送自描述的节点执行事件
                event_data = {
                    "type": "node_executed",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "node_id": node_name,
                        "state_update": str(state_update)[:100],  # 限制大小
                    }
                }
                
                # 特殊处理情绪检测
                if "detected_emotion" in state_update:
                    event_data["metadata"]["emotion"] = state_update["detected_emotion"]
                
                # 特殊处理最终回复
                if "final_response" in state_update:
                    full_response_buffer = state_update["final_response"]
                    event_data["metadata"]["response_preview"] = full_response_buffer[:50]
                
                await websocket.send_json(event_data)

            # 4. 发送完整回复
            await websocket.send_json({
                "type": "response_complete",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "response": full_response_buffer,
                }
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
                "metadata": {}
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "message": str(e)
                }
            })
        except:
            pass

# --- Uvicorn 启动配置 ---
if __name__ == "__main__":
    import uvicorn
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    uvicorn.run(app, host="0.0.0.0", port=8000)
