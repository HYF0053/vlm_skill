import json
import re
from typing import List, Optional, Generator
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .skills import SkillMiddleware

def create_dynamic_agent(provider: str, api_url: str, model_name: str, checkpointer, middleware: List, system_prompt: str = None, tools: List = None):
    """Creates a LangChain agent with the specified model configuration."""
    base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"
    api_key = "ollama" if provider == "Ollama" else "EMPTY"
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    if not system_prompt:
        system_prompt = (
            "You are an intelligent assistant with access to a library of capabilities (skills). "
            "Use them to help the user. "
        )

    # 如果有提供 tools 列表，則傳遞給 create_agent
    # 否則依賴 middleware 提供的工具
    agent = create_agent(
        llm,
        system_prompt=system_prompt,
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    return agent

def try_execute_text_tool_calls(content: str, log_lines: list, tool_registry: dict) -> tuple[bool, str]:
    """Detect and execute <tool_call> JSON blocks found in plain-text AI content."""
    pattern = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)
    matches = pattern.findall(content)
    if not matches:
        return False, content

    executed_any = False
    for raw_json in matches:
        try:
            call = json.loads(raw_json)
            tool_name = call.get("name") or call.get("function", {}).get("name", "")
            args = call.get("arguments") or call.get("parameters") or {}
            if isinstance(args, str): args = json.loads(args)

            log_lines.append(f"🔧 [Text tool_call] Calling Tool: {tool_name}")
            tool_fn = tool_registry.get(tool_name)
            if tool_fn:
                result = tool_fn.invoke(args)
                log_lines.append(f"✅ Tool Output ({tool_name}):\n{result}")
                executed_any = True
            else:
                log_lines.append(f"❌ Tool '{tool_name}' not found.")
        except Exception as e:
            log_lines.append(f"⚠️ Tool call error: {e}")

    remaining = pattern.sub("", content).strip()
    return executed_any, remaining

def run_agent_stream(agent, inputs, config, log_lines, tool_registry, chatbot_history, final_system_prompt="", usage=None, memory_params=None):
    """Generator to run agent and yield chatbot history, logs, system prompt, and usage HTML."""
    from utils.helpers import generate_usage_html, get_token_estimate
    current_output = ""
    _last_tool_sig = ""
    _dup_count = 0
    MAX_DUP_CALLS = 3
    
    usage = usage or {"total": 0}
    max_tokens = memory_params["max_model_len"] if memory_params else 4096
    cpt = memory_params["chars_per_token"] if memory_params else 1.8

    try:
        for update in agent.stream(inputs, config, stream_mode="updates"):
            for node_name, node_output in update.items():
                msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
                for msg in msgs:
                    msg_type = getattr(msg, "type", None)
                    if msg_type == "ai":
                        content = getattr(msg, "content", "") or ""
                        
                        # Capture actual usage if provided by metadata
                        meta = getattr(msg, "response_metadata", {})
                        if "token_usage" in meta:
                            # If the model provides exact counts, we prioritize them
                            actual = meta["token_usage"]
                            usage["output"] = actual.get("completion_tokens", usage.get("output", 0))
                        
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            name, args = tc.get("name", ""), tc.get("args", {})
                            sig = f"{name}::{repr(args)}"
                            if sig == _last_tool_sig: _dup_count += 1
                            else: _last_tool_sig, _dup_count = sig, 1
                            if _dup_count >= MAX_DUP_CALLS:
                                log_lines.append(f"⚠️ Repeated tool call {name} x{_dup_count}. Stopping.")
                                return
                            log_lines.append(f"🔧 Calling Tool: {name}")

                        if content:
                            if tool_calls: 
                                log_lines.append(f"🤖 AI Thinking:\n{content}")
                            else:
                                executed, remaining = try_execute_text_tool_calls(content, log_lines, tool_registry)
                                if not executed:
                                    log_lines.append(f"🤖 Final Answer:\n{content}")
                                    current_output = content
                                    # Fallback estimate for output tokens
                                    usage["output"] = get_token_estimate(len(current_output), cpt)
                                elif remaining: 
                                    log_lines.append(f"🤖 AI Thinking:\n{remaining}")
                        
                        # Update total
                        usage["total"] = sum(v for k, v in usage.items() if k != "total")
                        usage_html = generate_usage_html(usage, max_tokens)
                        
                        if current_output: 
                            chatbot_history[-1][1] = current_output
                        else:
                            chatbot_history[-1][1] = "🤖 Agent is thinking/acting..."
                        
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

                    elif msg_type == "tool":
                        content = getattr(msg, "content", "") or ""
                        log_lines.append(f"✅ Tool Output ({getattr(msg, 'name', 'tool')}):\n{content}")
                        
                        # Track tool result tokens
                        usage["tool_results"] += get_token_estimate(len(content), cpt)
                        usage["total"] = sum(v for k, v in usage.items() if k != "total")
                        usage_html = generate_usage_html(usage, max_tokens)
                        
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html
    except Exception as e:
        err = str(e)
        msg = "⚠️ Agent exceeded max steps." if "recursion" in err.lower() else f"❌ Error: {err}"
        log_lines.append(msg)
        chatbot_history[-1][1] = current_output or err
        usage_html = generate_usage_html(usage, max_tokens)
        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

    if not current_output:
        chatbot_history[-1][1] = "Finished without response."
    
    usage["total"] = sum(v for k, v in usage.items() if k != "total")
    usage_html = generate_usage_html(usage, max_tokens)
    yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html
