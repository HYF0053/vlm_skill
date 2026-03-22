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
            "IMPORTANT: Always respond in Traditional Chinese (正體中文). "
            "Provide the final answer directly in Traditional Chinese, excluding any thinking process, reasoning, or internal thoughts.\n\n"
            "When using skills and answering questions, please follow this retrieval priority order:\n"
            "1. Current Context (Short-term): Answer directly if the info is within the current conversation window (Note: The window only holds the last ~10 turns. Older details are dynamically trimmed).\n"
            "2. Agent Memory & User RAG (Long-term): If the context was trimmed or involves 'past decisions', 'project specs', or 'past workflows', you MUST search Long-term memory:\n"
            "   - Use `Memory Skill` (search_memo_qdrant.py) for past agent workflows and project memory.\n"
            "   - Use `RAG Skill` (search_vdb.py) for searching the user's external documents databases.\n"
            "3. MCP Servers (Real-time/External Data): When needing to read external services (e.g., GitHub, Slack) or perform specific actions.\n"
            "4. Web Search (External Online Knowledge): When internal memory has no results or the question involves general latest external technical knowledge."
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
    return agent, llm

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

def run_agent_stream(agent, inputs, config, log_lines, tool_registry, chatbot_history, final_system_prompt="", usage=None, memory_params=None, llm=None):
    """Generator to run agent and yield chatbot history, logs, system prompt, and usage HTML."""
    from utils.helpers import generate_usage_html, get_token_estimate
    current_output = ""
    _last_tool_sig = ""
    _dup_count = 0
    MAX_DUP_CALLS = 3
    _streaming_ai_buffer = ""   # accumulates token chunks during LLM streaming
    _streaming_log_idx = None   # index in log_lines for the live streaming entry

    usage = usage or {"total": 0}
    max_tokens = memory_params["max_model_len"] if memory_params else 4096
    cpt = memory_params["chars_per_token"] if memory_params else 1.8

    def _flush_streaming_buffer():
        """Remove the temporary streaming log entry when a full message arrives."""
        nonlocal _streaming_ai_buffer, _streaming_log_idx
        if _streaming_log_idx is not None and _streaming_log_idx < len(log_lines):
            log_lines.pop(_streaming_log_idx)
        _streaming_ai_buffer = ""
        _streaming_log_idx = None

    try:
        for event_type, event_data in agent.stream(inputs, config, stream_mode=["updates", "messages"]):

            # ── Token-level streaming from the LLM ──────────────────────────────
            if event_type == "messages":
                chunk, _meta = event_data if isinstance(event_data, tuple) else (event_data, {})
                chunk_content = getattr(chunk, "content", "") or ""
                if not chunk_content:
                    continue
                # Only accumulate AI message chunks (ignore tool result chunks)
                if getattr(chunk, "type", None) == "AIMessageChunk" or chunk.__class__.__name__ == "AIMessageChunk":
                    _streaming_ai_buffer += chunk_content
                    streaming_entry = f"🤖 AI Streaming...\n{_streaming_ai_buffer}"
                    if _streaming_log_idx is None:
                        log_lines.append(streaming_entry)
                        _streaming_log_idx = len(log_lines) - 1
                    else:
                        log_lines[_streaming_log_idx] = streaming_entry
                    usage_html = generate_usage_html(usage, max_tokens)
                    yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html
                continue

            # ── Node-level updates (complete messages) ───────────────────────────
            if event_type != "updates":
                continue

            update = event_data
            for node_name, node_output in update.items():
                msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
                for msg in msgs:
                    msg_type = getattr(msg, "type", None)
                    if msg_type == "ai":
                        # Remove the live streaming entry; we now have the complete message
                        _flush_streaming_buffer()

                        content = getattr(msg, "content", "") or ""
                        
                        # Capture actual usage if provided by metadata
                        meta = getattr(msg, "response_metadata", {})
                        if "token_usage" in meta:
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
                            # Filter out thinking process and keep only the final answer
                            filtered_content = content
                            
                            thinking_patterns = [
                                "太好了", "我找到了", "讓我", "根據", "從搜索結果", "我為您",
                                "根據搜尋結果", "讓我把", "整理成", "以下", "我看到", "我來查詢",
                                "好的", "我理解", "首先", "其次", "最後", "總結", "以下是"
                            ]
                            
                            lines = content.split('\n')
                            final_answer_lines = []
                            in_thinking = True
                            
                            for line in lines:
                                if in_thinking and (line.strip() == '' or line.strip().startswith('===') or line.strip().startswith('---')):
                                    in_thinking = False
                                if not in_thinking:
                                    final_answer_lines.append(line)
                                elif line.strip() and not any(line.strip().startswith(p) for p in thinking_patterns):
                                    final_answer_lines.append(line)
                            
                            if final_answer_lines:
                                filtered_content = '\n'.join(final_answer_lines).strip()
                            
                            if tool_calls:
                                log_lines.append(f"🤖 AI Thinking:\n{content}")
                            else:
                                executed, remaining = try_execute_text_tool_calls(filtered_content, log_lines, tool_registry)
                                if not executed:
                                    log_lines.append(f"🤖 Final Answer:\n{filtered_content}")
                                    current_output = filtered_content
                                    usage["output"] = get_token_estimate(len(current_output), cpt)
                                elif remaining:
                                    log_lines.append(f"🤖 AI Thinking:\n{remaining}")
                        
                        usage["total"] = sum(v for k, v in usage.items() if k != "total")
                        usage_html = generate_usage_html(usage, max_tokens)
                        
                        if current_output:
                            chatbot_history[-1][1] = current_output
                        else:
                            chatbot_history[-1][1] = "🤖 Agent is thinking/acting..."
                        
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

                        # Preemptive token limit check (90% threshold) to avoid hard provider errors
                        if usage["total"] >= max_tokens * 0.90:
                            raise Exception("Preemptive Token Limit")

                    elif msg_type == "tool":
                        _flush_streaming_buffer()
                        content = getattr(msg, "content", "") or ""
                        if len(content) > 3000:
                            content = content[:1500] + "\n...[TRUNCATED_DUE_TO_LENGTH]...\n" + content[-1500:]
                        log_lines.append(f"✅ Tool Output ({getattr(msg, 'name', 'tool')}):\n{content}")
                        
                        usage["tool_results"] += get_token_estimate(len(content), cpt)
                        usage["total"] = sum(v for k, v in usage.items() if k != "total")
                        usage_html = generate_usage_html(usage, max_tokens)
                        
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

                        if usage["total"] >= max_tokens * 0.90:
                            raise Exception("Preemptive Token Limit")

    except Exception as e:
        _flush_streaming_buffer()
        err = str(e).lower()
        is_recursion = "recursion" in err
        is_token_limit = any(k in err for k in ["preemptive token limit", "context length", "token limit", "maximum context length", "context_length_exceeded", "max_tokens"])
        
        if is_recursion:
            msg = "⚠️ Agent exceeded max steps."
        elif is_token_limit:
            msg = "⚠️ Agent approaching or exceeded token limit."
        else:
            msg = f"❌ Error: {str(e)}"
            
        log_lines.append(msg)
        
        if (is_recursion or is_token_limit) and llm:
            log_lines.append("🤖 Generating progress summary...")
            yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, generate_usage_html(usage, max_tokens)
            
            logs_text = "\n".join([str(l)[:1500] for l in log_lines[-20:]])
            summary_prompt = (
                "您是一個任務總結助手。AI Agent 在執行任務時達到了最大步數或 Token 限制，目前被迫停止。\n"
                "請根據以下的執行日誌（Logs），總結目前的進度、已經完成的事項、尚未完成的部分，並詢問用戶是否需要繼續（增加步數/清除部分記憶）或就此結束。\n\n"
                "執行日誌：\n"
                f"{logs_text}\n\n"
                "請務必使用 正體中文 (Traditional Chinese) 回答，並直接提供摘要（包含當前進度、成果、待辦事項及詢問用戶意向），不要包含任何開場白。"
            )
            try:
                summary_res = llm.invoke(summary_prompt)
                current_output = summary_res.content
            except Exception as sum_err:
                current_output = f"⚠️ 達到步數或 Token 限制，且摘要生成失敗: {sum_err}"
        else:
            current_output = current_output or str(e)

        chatbot_history[-1][1] = current_output
        usage_html = generate_usage_html(usage, max_tokens)
        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

    _flush_streaming_buffer()
    if not current_output:
        chatbot_history[-1][1] = "Finished without response."
    
    usage["total"] = sum(v for k, v in usage.items() if k != "total")
    usage_html = generate_usage_html(usage, max_tokens)
    yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html
