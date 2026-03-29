import os
import gradio as gr
from utils.helpers import classify_uploaded_file, encode_image, get_ollama_models, get_vllm_models, compute_memory_params, get_token_estimate
from core.agent import create_dynamic_agent, run_agent_stream
from core.memory import ThreadMemory
from langchain_openai import ChatOpenAI

class UIHandler:
    def __init__(self, memory_store, skill_repo, tool_registry, global_checkpointer):
        self.memory_store = memory_store
        self.skill_repo = skill_repo
        self.tool_registry = tool_registry
        self.global_checkpointer = global_checkpointer

    def get_live_logs(self):
        """Reads the live log file for real-time tool output display."""
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs", "live.log")
        if not os.path.exists(log_path):
            return "No active tool execution logs found."
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                # Return last 50 lines to keep it readable
                lines = f.readlines()
                return "".join(lines[-50:])
        except Exception as e:
            return f"Error reading logs: {e}"

    def refresh_models(self, provider, api_url):
        if not api_url: return gr.Dropdown(choices=[])
        models = get_ollama_models(api_url) if provider == "Ollama" else get_vllm_models(api_url)
        if not models:
            gr.Info(f"No models found at {api_url}")
            return gr.Dropdown(choices=[], value=None)
        return gr.Dropdown(choices=models, value=models[0])

    def refresh_asr_models(self, asr_url):
        """從 ASR API 取得可用模型列表。"""
        if not asr_url:
            return gr.Dropdown(choices=[])
        models = get_vllm_models(asr_url)
        if not models:
            gr.Info(f"No ASR models found at {asr_url}")
            return gr.Dropdown(choices=[], value=None)
        return gr.Dropdown(choices=models, value=models[0])

    def run_agent_task(self, file_upload, provider, api_url, model_name, image_max_size, max_agent_steps, system_prompt, user_prompt, chatbot_history, session_key, asr_url=None, asr_model=None):
        session_key = str(session_key)
        print(f"[UIHandler] Running task for session: {session_key}, query: {user_prompt[:50]}...")
        from utils.helpers import get_model_context_len
        if chatbot_history is None: chatbot_history = []
        query = user_prompt.strip() or "請問有什麼我可以幫助您？"
        log_lines = []
        final_system_prompt = ""
        usage_html = ""
        
        try:
            if not model_name:
                chatbot_history.append((query, "Please select a model."))
                yield chatbot_history, "", "", "", gr.update(), gr.update()
                return

            file_type = classify_uploaded_file(file_upload)
            file_name = os.path.basename(file_upload) if file_upload else ""
            if file_type == 'image':
                user_display = f"🖼️ [{file_name}]\n\n{query}" if file_upload else query
            elif file_type == 'audio':
                user_display = f"🎙️ [{file_name}]\n\n{query}" if file_upload else query
            else:
                user_display = f"📄 [{file_name}]\n\n{query}" if file_upload else query

            # Memory & Context Config
            ctx_len = get_model_context_len(provider, api_url, model_name)
            if ctx_len:
                memory_params = compute_memory_params(ctx_len, image_tokens_reserve=2048 if file_type=='image' else 0)
                print(f"[UIHandler] Detected context length: {ctx_len}, Memory Params: {memory_params}")
            else:
                memory_params = None
                print(f"[UIHandler] Could not detect context length for {model_name}, using defaults.")

            # 1. Initialize Skill Middleware with Memory Store for structured prompt injection
            from core.skills import SkillMiddleware
            mw = SkillMiddleware(self.skill_repo, memory_store=self.memory_store, session_key=session_key)
            
            # The tools are now managed by the middleware, including the new 'upsert_memory'
            tools_list = mw.tools
            
            # 2. Get the structured memory context (User Profile, Facts, etc.)
            try:
                from core.memory import format_memory_for_prompt
                mem = self.memory_store.load_thread(session_key)
                structured_context = format_memory_for_prompt(mem)
            except Exception:
                structured_context = "\n\n(Memory system unavailable.)\n"
            
            import datetime
            current_time_info = (
                f"\n\n[System Info]\n"
                f"Current Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Day of the Week: {datetime.datetime.now().strftime('%A')}\n"
            )

            conflict_rule = (
                "\n\n🚨 CRITICAL MEMORY RULE: The STRUCTURED MEMORY block above represents the ABSOLUTE TRUTH of the user's CURRENT status and preferences. "
                "If past conversation history contradicts the Structured Memory (e.g. user changed their mind later), you MUST IGNORE the history and STRICTLY OBEY the Structured Memory. "
                "If a user's preference is ALREADY recorded or up-to-date in the Structured Memory, DO NOT call `upsert_memory` again for it."
            )
            base_system_prompt = (system_prompt or "") + current_time_info + structured_context + conflict_rule

            # Use the middleware's unified addendum logic for UI transparency
            skills_addendum = mw.get_skills_addendum()
            final_system_prompt = base_system_prompt + skills_addendum

            # Calculate current usage breakdown (Estimated)
            from utils.helpers import generate_usage_html
            cpt = memory_params["chars_per_token"] if memory_params else 1.8
            
            input_prompt = get_token_estimate(len(query), cpt)
            input_context = get_token_estimate(len(final_system_prompt), cpt)
            
            # Inject dynamic token limits for this model (Paths are already global via app.py)
            os.environ["MAX_MODEL_LEN"] = str(ctx_len or 4096)
            os.environ["CHARS_PER_TOKEN"] = str(memory_params.get("chars_per_token", 1.8) if memory_params else 1.8)
            
            # 取代 LangGraph checkpointer 的殘留狀態，改為使用我們自訂的 Smart Trimmed History
            try:
                input_history = 0
                formatted_history = []
                if hasattr(mem, "recent_messages") and mem.recent_messages:
                    for m in mem.recent_messages:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        
                        # 如果該輪有附件（如圖片、文本或PDF），重新載入給模型看，確保「長時視覺」不中斷
                        attachments = m.get("attachments", [])
                        if attachments:
                            msg_list = [{"type": "text", "text": content}]
                            for att in attachments:
                                att_path = att.get("path", "")
                                if not os.path.exists(att_path):
                                    continue
                                
                                att_type = att.get("type")
                                if att_type == "image":
                                    b64 = encode_image(att_path, int(image_max_size))
                                    if b64:
                                        msg_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                                elif att_type == "text_snippet":
                                    try:
                                        with open(att_path, "r", encoding="utf-8", errors="replace") as f:
                                            snippet = f.read()
                                        file_name = os.path.basename(att_path)
                                        msg_list[0]["text"] += f"\n\n檔案 [{file_name}] 內容：\n```\n{snippet[:8000]}\n```"
                                    except Exception:
                                        pass
                                elif att_type == "other":
                                    file_name = os.path.basename(att_path)
                                    msg_list[0]["text"] += f"\n\n請處理檔案：**{file_name}** (`{att_path}`)"
                            formatted_history.append({"role": role, "content": msg_list})
                        else:
                            formatted_history.append({"role": role, "content": content})
                    
                    actual_history_len = sum(len(str(m.get("content", ""))) for m in mem.recent_messages)
                    input_history = get_token_estimate(actual_history_len, cpt)
                    
                    # 加上歷史圖片與檔案的 Token 預估
                    for m in mem.recent_messages:
                        for att in m.get("attachments", []):
                            if att.get("type") == "image":
                                input_history += 1000
                            elif att.get("type") == "text_snippet":
                                input_history += 1000
                            elif att.get("type") == "other":
                                input_history += 50
            except Exception as e:
                print(f"[UIHandler] History formatting error: {e}")
                input_history = 0
            
            usage = {
                "input_prompt": input_prompt,
                "input_context": input_context,
                "input_history": input_history,
                "output": 0,
                "tool_results": 0,
                "total": input_prompt + input_context + input_history
            }
            
            # 將 LLM provider 設定同步寫入環境變數，讓子進程（execute_script / run_python_code）繼承
            base_url_v1 = api_url.rstrip("/") + ("/v1" if not api_url.endswith("/v1") else "")
            os.environ["VLLM_BASE_URL"] = base_url_v1
            os.environ["VLLM_MODEL"]    = model_name
            os.environ["VLLM_API_KEY"]  = "ollama" if provider == "Ollama" else "EMPTY"
            os.environ["SESSION_ID"]    = session_key
            if memory_params:
                os.environ["MAX_MODEL_LEN"] = str(memory_params["max_model_len"])
                os.environ["CHARS_PER_TOKEN"] = str(memory_params["chars_per_token"])
            if asr_url:
                os.environ["ASR_API_URL"] = asr_url.rstrip("/")
            if asr_model:
                os.environ["ASR_MODEL"] = asr_model

            # Agent preparation (SkillMiddleware will handle the system prompt injection in LangGraph)
            agent, llm = create_dynamic_agent(provider, api_url, model_name, self.global_checkpointer, [mw], system_prompt, tools_list)
            message_content = [{"type": "text", "text": query}]
            current_attachments = []
            if file_upload:
                if file_type == 'image':
                    b64 = encode_image(file_upload, int(image_max_size))
                    if b64: 
                        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        usage["input_context"] += 1000 
                        current_attachments.append({"type": "image", "path": file_upload})
                elif file_type == 'text':
                    with open(file_upload, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    text_blob = f"\n\n檔案 [{file_name}] 內容：\n```\n{content[:8000]}\n```"
                    message_content[0]["text"] += text_blob
                    usage["input_context"] += get_token_estimate(len(text_blob), cpt)
                    current_attachments.append({"type": "text_snippet", "path": file_upload})
                elif file_type == 'audio':
                    _asr_url_hint = asr_url or os.environ.get('ASR_API_URL', 'http://localhost:8000')
                    _asr_model_hint = asr_model or os.environ.get('ASR_MODEL', '')
                    model_arg = f"--model {_asr_model_hint}" if _asr_model_hint else ""
                    message_content[0]["text"] += (
                        f"\n\n用戶上傳了音頻檔案：**{file_name}** (`{file_upload}`)\n"
                        f"請使用 ASR skill 進行語音轉文字，執行指令範例：\n"
                        f"`python skills/asr/scripts/transcribe.py --audio_path {file_upload} --asr_url {_asr_url_hint} {model_arg}`\n"
                        f"（也可透過 execute_script 呼叫，skill_name=asr, script_path=scripts/transcribe.py）"
                    )
                    current_attachments.append({"type": "other", "path": file_upload})
                else:
                    message_content[0]["text"] += f"\n\n請處理檔案：**{file_name}** (`{file_upload}`)"
                    current_attachments.append({"type": "other", "path": file_upload})

            usage["total"] = sum(v for k, v in usage.items() if k != "total")
            usage_html = generate_usage_html(usage, memory_params["max_model_len"] if memory_params else 4096)
            
            
            chatbot_history.append([user_display, "⏳ Initializing Agent..."])
            log_lines = ["⏳ Initializing Agent...", "⏳ Running Inference..."]
            
            yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html, gr.update(), gr.update()

            import uuid
            ephemeral_thread_id = f"{session_key}_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": ephemeral_thread_id}, "recursion_limit": int(max_agent_steps)}
            
            messages_payload = formatted_history + [{"role": "user", "content": message_content}]
            
            # Run stream
            gen = run_agent_stream(agent, {"messages": messages_payload}, config, log_lines, self.tool_registry, chatbot_history, final_system_prompt, usage=usage, memory_params=memory_params, llm=llm)
            for hist, logs, sys_prompt, usage_html in gen:
                yield hist, logs, sys_prompt, usage_html, gr.update(), gr.update()

            # Memory save
            final_answer = chatbot_history[-1][1]

            # ── Inline turn metadata extraction (zero LLM calls) ──────────────
            try:
                from core.turn_meta import extract_turn_meta
                turn_meta = extract_turn_meta(query, final_answer)
            except Exception as _tm_err:
                turn_meta = None
                print(f"[UIHandler] turn_meta extraction failed: {_tm_err}")
            # ──────────────────────────────────────────────────────────────────

            try:
                # 更新長期記憶
                updated_mem = None
                gen_mem = self.memory_store.record_turn(session_key, query, final_answer, attachments=current_attachments, usage=usage, turn_meta=turn_meta)
                
                try:
                    while True:
                        status = next(gen_mem)
                        log_lines.append(f"🧠 {status}")
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html, gr.update(), gr.update()
                except StopIteration as e:
                    updated_mem = e.value
                
                if updated_mem:
                    updated_mem.usage = usage
                    self.memory_store.save_thread(updated_mem)
                            
                # ─── 將目前的記憶狀態列在 Log 最下方 ───
                final_mem = updated_mem if updated_mem else self.memory_store.load_thread(session_key)
                if final_mem:
                    log_lines.append("\n" + "="*40)
                    log_lines.append("🗂️ 【當前被載入 RAM 的歷史記憶清單】")
                        
                    for m in final_mem.recent_messages:
                        role = "👤 [User]" if m.get("role") == "user" else "🤖 [AI]"
                        preview = m.get('content', '')
                        if len(preview) > 100: preview = preview[:100] + "..."
                        
                        log_lines.append(f"{role} {m.get('ts', '')[:19].replace('T', ' ')}:\n{preview}\n")
                    log_lines.append("="*40)

                yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html, None, ""
            except Exception as e:
                log_lines.append(f"⚠️ Memory error: {e}")
                yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html, None, ""

        except Exception as e:
            log_lines.append(f"❌ Error: {e}")
            yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html, None, ""

    # --- Skill Editor Handlers ---
    def on_skill_select(self, skill_name):
        if not skill_name: return gr.Dropdown(choices=[], value=None), ""
        files = self.skill_repo.list_skill_files(skill_name)
        default_file = "SKILL.md" if "SKILL.md" in files else (files[0] if files else None)
        content = self.skill_repo.get_skill_details(skill_name, default_file) if default_file else ""
        return gr.Dropdown(choices=files, value=default_file), content

    def save_skill_content(self, skill_name, file_path, new_content):
        skill = self.skill_repo._find_skill_by_name(skill_name)
        if not skill or not file_path: return "Error."
        target_path = os.path.join(skill.path, file_path)
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f: f.write(new_content)
            return f"Saved {file_path}"
        except Exception as e: return f"Error: {e}"

    # --- Skill Refresh Handler ---
    def refresh_skills_list(self):
        """清除 skill 快取並重新載入"""
        self.skill_repo._cache_skills = None
        return [s.name for s in self.skill_repo.get_all_skills()]

    # --- Memory Manager Handlers ---
    def list_memories(self):
        """列出所有記憶線程"""
        threads = self.memory_store.list_threads()
        if not threads:
            return [["(尚無記憶)", "-", "-", "-", "-"]]
        rows = []
        for t in threads:
            msg_count    = len(t.recent_messages)
            deleted_flag = "🗑️ 已刪除" if t.is_deleted else "✅ 使用中"
            label        = t.display_name.strip() if t.display_name.strip() else t.session_key
            rows.append([
                t.session_key,
                str(t.turn_count),
                str(msg_count),
                t.last_updated_at[:16].replace("T", " "),
                deleted_flag,
            ])
        return rows

    def inspect_memory(self, key):
        """查詢特定 Session Key 的記憶"""
        if not key:
            return "(請輸入 Session Key)"
        mem = self.memory_store.load_thread(key.strip())
        if not mem.recent_messages and not mem.turn_count:
            return f"找不到 key='{key.strip()}' 的記憶，或該 key 尚無內容。"

        from core.memory import load_global_profile
        profile = load_global_profile()

        lines = [
            f"🔑 Session Key: {mem.session_key}",
            f"📊 總輪數: {mem.turn_count}",
            f"🗑️ 已刪除: {'是' if mem.is_deleted else '否'}",
            f"🕐 建立: {mem.created_at[:16]}　更新: {mem.last_updated_at[:16]}",
            "",
            "─── 👤 用戶檔案 (global_profile.json) ───",
            str(profile.get('user_profile', {})) or "(無)",
            "",
            "─── ⚙️ 用戶偏好 (global_profile.json) ───",
            str(profile.get('preferences', {})) or "(無)",
            "",
            "─── 📜 AI 行動準則 (global_profile.json) ───",
            str(profile.get('agent_rules', {})) or "(無)",
            "",
            "───  最近幾輪對話 ───",
        ]
        for m in mem.recent_messages[-6:]:
            role = "👤 使用者" if m.get("role") == "user" else "🤖 AI"
            content = m.get("content", "")
            qdrant_tag = f" [Qdrant: {m.get('qdrant_collection','')} {m.get('qdrant_id','')[:8]}]" if m.get('qdrant_id') else ""
            lines.append(f"{role}: {content[:200]}{'...' if len(content)>200 else ''}{qdrant_tag}")
        return "\n".join(lines)

    def delete_memory(self, key, provider=None, api_url=None, model_name=None):
        """刪除特定 Session Key 的記憶，並觸發背景 archive 總結。"""
        if not key:
            return "請輸入要刪除的 Session Key。"
        llm = None
        if provider and api_url and model_name:
            try:
                from langchain_openai import ChatOpenAI
                base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"
                api_key  = "ollama" if provider == "Ollama" else "EMPTY"
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0)
            except Exception as e:
                print(f"[delete_memory] LLM init failed: {e}")
        deleted = self.memory_store.delete_thread(key.strip(), llm=llm)
        if deleted:
            suffix = " (背景 archive 總結已啟動)" if llm else " (未提供 LLM，跳過 archive 總結)"
            return f"✅ 已刪除 key='{key.strip()}' 的記憶。{suffix}"
        return f"⚠️ 找不到 key='{key.strip()}'，無需刪除。"

    def get_usage_status(self, session_key, provider, api_url, model_name):
        """用於在切換 Session 或模型時即時更新 Token 使用量顯示"""
        from utils.helpers import generate_usage_html, get_model_context_len
        
        if not session_key or not model_name:
            return "<div style='color: gray; font-size: 14px;'>選擇模型與 Session 後將顯示 Token 使用量...</div>"
        
        mem = self.memory_store.load_thread(session_key)
        ctx_len = get_model_context_len(provider, api_url, model_name) or 4096
        
        # 如果該 session 尚無 usage 記錄，初始化一個空的
        usage = getattr(mem, "usage", {
            "input_prompt": 0,
            "input_context": 0,
            "input_history": 0,
            "output": 0,
            "tool_results": 0,
            "total": 0
        })
        
        return generate_usage_html(usage, ctx_len)
    # --- Session Management Handlers ---
    def on_session_change(self, session_key, provider=None, api_url=None, model_name=None):
        """當切換 Session 時，更新對話記錄框與 Token 使用量顯示"""
        print(f"[UIHandler] Session changed to: {session_key}")
        if not session_key:
            return [], "<div style='color: gray; font-size: 14px;'>請選擇或建立一個 Session...</div>"
        
        if not (provider and api_url and model_name):
            return [], "<div style='color: gray; font-size: 14px;'>選擇模型與 Session 後將顯示 Token 使用量...</div>"

        try:
            history = self._get_chatbot_history(session_key)
            usage_html = self.get_usage_status(session_key, provider, api_url, model_name)
            return history, usage_html
        except Exception as e:
            print(f"[UIHandler] Error in on_session_change: {e}")
            return [], f"錯誤: {e}"

    def on_add_session_simple(self):
        """建立新 Session（UUID key）"""
        session_key = self.memory_store.get_next_session_name()
        from core.memory import ThreadMemory
        self.memory_store.save_thread(ThreadMemory(session_key=session_key))
        keys = self.memory_store.list_session_keys()
        return gr.Dropdown(choices=keys, value=session_key)

    def on_delete_session_simple(self, session_key, provider=None, api_url=None, model_name=None):
        """刪除 Session（含磁碟記憶與 RAM checkpointer），觸發背景 archive 總結。"""
        # Clear InMemorySaver RAM checkpoints
        try:
            cp = self.global_checkpointer
            if hasattr(cp, 'storage'):
                keys_to_del = [k for k in cp.storage if (isinstance(k, tuple) and k[0] == session_key) or k == session_key]
                for k in keys_to_del:
                    del cp.storage[k]
            if hasattr(cp, 'writes'):
                keys_to_del = [k for k in cp.writes if (isinstance(k, tuple) and k[0] == session_key) or k == session_key]
                for k in keys_to_del:
                    del cp.writes[k]
        except Exception as e:
            print(f"[on_delete_session_simple] Checkpointer clear error: {e}")

        if not session_key:
            choices = self.memory_store.list_session_choices()
            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), []

        # Build LLM for archive summarization
        llm = None
        if provider and api_url and model_name:
            try:
                from langchain_openai import ChatOpenAI
                base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"
                api_key  = "ollama" if provider == "Ollama" else "EMPTY"
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0)
            except Exception as e:
                print(f"[on_delete_session_simple] LLM init failed: {e}")

        self.memory_store.delete_thread(session_key, llm=llm)

        keys = self.memory_store.list_session_keys()
        if keys:
            return gr.Dropdown(choices=keys, value=keys[0]), []
        else:
            return gr.Dropdown(choices=[], value=None), []

    def _get_chatbot_history(self, session_key):
        """Helper: 將 MemoryStore 的格式轉為 Gradio Chatbot 格式 [[user, ai], ...]"""
        mem = self.memory_store.load_thread(str(session_key))
        history = []
        msgs = mem.recent_messages
        for i in range(0, len(msgs), 2):
            u = msgs[i].get("content", "") if i < len(msgs) else ""
            a = msgs[i+1].get("content", "") if (i+1) < len(msgs) else ""
            if u or a:
                history.append([u, a])
        return history

    def _format_history_table(self, mem) -> list:
        if not mem: return []
        data = []
        for m in mem.recent_messages:
            preview = m.get("content", "")
            if len(preview) > 60: preview = preview[:60] + "..."
            ts = m.get("ts", "")[:19].replace("T", " ") if m.get("ts") else ""
            role = "👤 User" if m.get("role") == "user" else "🤖 AI"
            data.append([role, preview, ts])
        return data
