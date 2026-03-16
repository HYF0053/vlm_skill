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

    def refresh_models(self, provider, api_url):
        if not api_url: return gr.Dropdown(choices=[])
        models = get_ollama_models(api_url) if provider == "Ollama" else get_vllm_models(api_url)
        if not models:
            gr.Info(f"No models found at {api_url}")
            return gr.Dropdown(choices=[], value=None)
        return gr.Dropdown(choices=models, value=models[0])

    def run_agent_task(self, file_upload, provider, api_url, model_name, image_max_size, max_agent_steps, system_prompt, user_prompt, chatbot_history, session_key):
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
                yield chatbot_history, "", "", ""
                return

            file_type = classify_uploaded_file(file_upload)
            file_name = os.path.basename(file_upload) if file_upload else ""
            user_display = f"{'🖼️' if file_type=='image' else '📄'} [{file_name}]\n\n{query}" if file_upload else query

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
            # This is now handled inside SkillMiddleware.wrap_model_call, 
            # but for UI visibility we can still construct it here if we want to show 'final_system_prompt'
            structured_context = self.memory_store.get_session_system_context(session_key)
            base_system_prompt = (system_prompt or "") + structured_context

            # Replicate the middleware's addendum logic for UI transparency
            skills_addendum = (
                "\n\n"
                "================================================================\n"
                "KNOWLEDGE SKILL LIBRARY (READ-ONLY REFERENCE — NOT CALLABLE)\n"
                "================================================================\n"
                f"{mw.skills_prompt}\n\n"
                "----------------------------------------------------------------\n"
                "CALLABLE TOOLS (all tools you may invoke directly):\n"
                "  - upsert_memory(key, value, mem_type='fact')\n"
                "  - load_skill_overview(skill_name)\n"
                "  - read_skill_file(skill_name, file_path)\n"
                "  - execute_script(skill_name, script_path, script_args)\n"
                "  - run_cli_command(command, working_directory)\n"
                "  - run_python_code(code, working_directory)\n"
                "================================================================"
            )
            final_system_prompt = base_system_prompt + skills_addendum

            # Calculate current usage breakdown (Estimated)
            from utils.helpers import generate_usage_html
            current_mem = self.memory_store.load_thread(session_key)
            cpt = memory_params["chars_per_token"] if memory_params else 1.8
            
            input_prompt = get_token_estimate(len(query), cpt)
            input_context = get_token_estimate(len(final_system_prompt), cpt)
            input_history = get_token_estimate(sum(len(m.get("content", "")) for m in current_mem.recent_messages), cpt)
            
            usage = {
                "input_prompt": input_prompt,
                "input_context": input_context,
                "input_history": input_history,
                "output": 0,
                "tool_results": 0,
                "total": input_prompt + input_context + input_history
            }
            
            # Agent preparation (SkillMiddleware will handle the system prompt injection in LangGraph)
            agent = create_dynamic_agent(provider, api_url, model_name, self.global_checkpointer, [mw], system_prompt, tools_list)
            message_content = [{"type": "text", "text": query}]
            if file_upload:
                if file_type == 'image':
                    b64 = encode_image(file_upload, int(image_max_size))
                    if b64: 
                        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        usage["input_context"] += 1000 
                elif file_type == 'text':
                    with open(file_upload, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    text_blob = f"\n\n檔案 [{file_name}] 內容：\n```\n{content[:8000]}\n```"
                    message_content[0]["text"] += text_blob
                    usage["input_context"] += get_token_estimate(len(text_blob), cpt)
                else:
                    message_content[0]["text"] += f"\n\n請處理檔案：**{file_name}** (`{file_upload}`)"

            usage["total"] = sum(v for k, v in usage.items() if k != "total")
            usage_html = generate_usage_html(usage, memory_params["max_model_len"] if memory_params else 4096)
            
            chatbot_history.append([user_display, "⏳ Initializing Agent..."])
            log_lines = ["⏳ Initializing Agent...", "⏳ Running Inference..."]
            yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

            config = {"configurable": {"thread_id": session_key}, "recursion_limit": int(max_agent_steps)}
            
            # Run stream
            gen = run_agent_stream(agent, {"messages": [{"role": "user", "content": message_content}]}, config, log_lines, self.tool_registry, chatbot_history, final_system_prompt, usage=usage, memory_params=memory_params)
            for hist, logs, sys_prompt, usage_html in gen:
                yield hist, logs, sys_prompt, usage_html

            # Memory save
            final_answer = chatbot_history[-1][1]
            try:
                # Simple turn record (No automatic summarization)
                updated_mem = self.memory_store.record_turn(session_key, query, final_answer)
                log_lines.append("🧠 Long-term memory turn recorded.")
                
                if updated_mem:
                    updated_mem.usage = usage
                    self.memory_store.save_thread(updated_mem)
                
                yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html
            except Exception as e:
                log_lines.append(f"⚠️ Memory error: {e}")
                yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

        except Exception as e:
            log_lines.append(f"❌ Error: {e}")
            yield chatbot_history, "\n\n".join(log_lines), final_system_prompt, usage_html

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
            # Create a preview from facts if available
            preview = ""
            if t.facts:
                preview = f"Facts: {len(t.facts)} | " + (t.facts[-1][:50] + "...") if t.facts else ""
            elif t.preferences:
                preview = f"Prefs: {len(t.preferences)}"
            else:
                preview = "(無結構化資料)"
                
            rows.append([
                t.session_key,
                str(t.turn_count),
                str(len(t.facts) + len(t.preferences)),
                t.last_updated_at[:16].replace("T", " "),
                preview,
            ])
        return rows

    def inspect_memory(self, key):
        """查詢特定 Session Key 的記憶"""
        if not key:
            return "(請輸入 Session Key)"
        mem = self.memory_store.load_thread(key.strip())
        if not mem.recent_messages and not mem.facts and not mem.preferences:
            return f"找不到 key='{key.strip()}' 的記憶，或該 key 尚無內容。"
        
        lines = [
            f"🔑 Session Key: {mem.session_key}",
            f"📊 總輪數: {mem.turn_count}",
            f"🕐 建立: {mem.created_at[:16]}　更新: {mem.last_updated_at[:16]}",
            "",
            "─── 👤 用戶檔案 (User Profile) ───",
            str(mem.user_profile) if mem.user_profile else "(無)",
            "",
            "─── ⚙️ 用戶偏好 (Preferences) ───",
            str(mem.preferences) if mem.preferences else "(無)",
            "",
            "─── 🚀 專案狀態 (Project Status) ───",
            str(mem.current_project_status) if mem.current_project_status else "(無)",
            "",
            "─── 💡 重要事實 (Facts) ───",
            "\n".join(f"- {f}" for f in mem.facts) if mem.facts else "(無)",
            "",
            "─── 💬 最近幾輪對話 ───",
        ]
        for m in mem.recent_messages[-6:]:
            role = "👤 使用者" if m.get("role") == "user" else "🤖 AI"
            content = m.get("content", "")
            lines.append(f"{role}: {content[:200]}{'...' if len(content)>200 else ''}")
        return "\n".join(lines)

    def delete_memory(self, key):
        """刪除特定 Session Key 的記憶"""
        if not key:
            return "請輸入要刪除的 Session Key。"
        deleted = self.memory_store.delete_thread(key.strip())
        if deleted:
            return f"✅ 已刪除 key='{key.strip()}' 的記憶。"
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
        
        try:
            # 1. 載入對話歷史
            history = self._get_chatbot_history(session_key)
            
            # 2. 更新 Token 使用量 (如果資訊充足)
            usage_html = ""
            if provider and api_url and model_name:
                usage_html = self.get_usage_status(session_key, provider, api_url, model_name)
            else:
                usage_html = "<div style='color: gray; font-size: 14px;'>對話已載入位，選擇模型後將顯示 Token 使用量...</div>"
                
            return history, usage_html
        except Exception as e:
            print(f"[UIHandler] Error in on_session_change: {e}")
            return [], f"錯誤: {e}"

    def on_add_session_simple(self):
        """新增一個 Session 並由 dropdown.change 觸發後續"""
        new_name = self.memory_store.get_next_session_name()
        from core.memory import ThreadMemory
        self.memory_store.save_thread(ThreadMemory(session_key=new_name))
        keys = self.memory_store.list_session_keys()
        return gr.Dropdown(choices=keys, value=new_name)

    def on_delete_session_simple(self, session_key):
        """刪除 Session 並由 dropdown.change 觸發後續"""
        if not session_key:
            return gr.Dropdown()
        
        self.memory_store.delete_thread(session_key)
        keys = self.memory_store.list_session_keys()
        if not keys:
            keys = ["main"]
            from core.memory import ThreadMemory
            self.memory_store.save_thread(ThreadMemory(session_key="main"))
            
        return gr.Dropdown(choices=keys, value=keys[0])

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
