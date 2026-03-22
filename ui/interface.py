import gradio as gr
# from core.memory import SESSION_KEY_PRESETS

def create_ui(handler):
    with gr.Blocks(title="Agentic Studio", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 🤖 Agentic Studio")
        
        with gr.Tabs():
            # --- TAB 1: Agent ---
            with gr.Tab("🤖 Agent"):
                with gr.Group():
                    gr.Markdown("### 🔑 Session Management")
                    initial_choices = handler.memory_store.list_session_keys() or ["main"]
                    if not handler.memory_store.list_session_keys():
                        from core.memory import ThreadMemory
                        handler.memory_store.save_thread(ThreadMemory(session_key="main"))

                    initial_session = initial_choices[0]

                    with gr.Row():
                        session_dropdown = gr.Dropdown(
                            label="Select Session", 
                            choices=initial_choices, 
                            value=initial_session,
                            scale=4,
                            interactive=True,
                            allow_custom_value=True
                        )
                        add_session_btn = gr.Button("➕ New", scale=1)
                        delete_session_btn = gr.Button("🗑️ Del", scale=1, variant="stop")
                    visual_usage_status = gr.HTML(value="<div style='color: gray; font-size: 14px;'>選擇模型後將顯示 Token 使用量...</div>")

                chatbot = gr.Chatbot(
                    label="Chat History", 
                    value=handler._get_chatbot_history(initial_session),
                    height=500
                )

                with gr.Accordion("📝 System Prompt Settings", open=False):
                    system_prompt_input = gr.TextArea(
                        label="System Prompt", 
                        value=(
                            "You are an intelligent assistant with access to a library of capabilities (skills). "
                            "Use them to help the user. IMPORTANT: Always respond in Traditional Chinese (正體中文). "
                            "Provide only the final answer directly in Traditional Chinese, excluding any thinking process or internal reasoning. "
                            "CRITICAL MEMORY RULE: 你具備「主動反思」的核心動能。在每一輪對話中，請主動捕捉具備長期價值的『新事實、用戶偏好或決策摘要』，並主動調用對應的「技能 (Skill)」進行結構化儲存。請務必在確實執行成功後，才在回覆中宣告已完成儲存，確保你的回覆精確反映實際操作。\n\n"
                            "When using skills and answering questions, please follow this retrieval priority order:\n"
                            "1. Current Context (Short-term): If the information is within the current conversation window, answer directly.\n"
                            "2. Memory (Personal/Project Knowledge):\n"
                            "   Trigger Conditions: Involving 'past discussions', 'user preferences', or 'project specifications'. Use this to recall how we worked together before.\n\n"
                            "3. RAG (Product/Technical Specs):\n"
                            "   Trigger Conditions: Involving user's 'product equipment information' or 'industry regulations/specifications'. Use this to look up technical facts.\n\n"
                            "4. MCP Servers (Real-time/External Data):\n"
                            "   Trigger Conditions: When needing to read external services (e.g., GitHub PR, Slack messages) or perform specific actions.\n\n"
                            "5. Web Search (External Online Knowledge):\n"
                            "   Trigger Conditions: When internal knowledge has no results or the question involves general latest external technical knowledge."
                        ),
                        lines=10
                    )

                user_prompt_input = gr.Textbox(label="💬 User Query", placeholder="Enter your command... (Press Enter to send)", lines=3)
                
                with gr.Row():
                    file_upload_input = gr.File(label="📎 Upload File", file_count="single")
                    with gr.Column():
                        run_btn = gr.Button("🚀 Run Agent", variant="primary", scale=2)
                        stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
                        clear_btn = gr.Button("🧹 Clear Chat", scale=1)

                with gr.Accordion("📜 Execution Logs & Technical Details", open=True):
                    log_output = gr.Textbox(label="Thinking Process & Logs", lines=10, max_lines=15, interactive=False)
                    with gr.Group():
                        gr.Markdown("#### ⚡ Live Command Output (Updates every 2s)")
                        live_log_view = gr.Code(label="Real-time Tool Output", language="markdown", lines=5, interactive=False)
                        refresh_timer = gr.Timer(value=2, active=True)
                        refresh_timer.tick(handler.get_live_logs, outputs=live_log_view)
                    
                    injected_prompt_output = gr.Textbox(label="Actual Injected System Prompt (ReadOnly)", lines=12, interactive=False)

            # --- TAB 2: Settings ---
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        provider_radio = gr.Radio(["vLLM", "Ollama"], label="LLM Provider", value="vLLM")
                        api_url_input = gr.Textbox(label="API URL", value="http://10.1.1.7:24131")
                        refresh_models_btn = gr.Button("🔄 Refresh Models")
                        model_dropdown = gr.Dropdown(label="Select Model", choices=[], interactive=True, allow_custom_value=True)
                    with gr.Column():
                        image_max_size_slider = gr.Slider(512, 4096, value=1024, step=512, label="Max Image Dimension")
                        max_agent_steps_slider = gr.Slider(5, 50, value=15, step=1, label="Max Agent Steps")

            # --- TAB 3: Memory Manager ---
            with gr.Tab("🧠 Memory Manager"):
                gr.Markdown(
                    "## 🧠 結構化持久記憶管理\n\n"
                    "查看所有已儲存的 Session 記憶。現在改為由 AI 自行決定要記住哪些事實與偏好。\n\n"
                    "📁 存放路徑：`data/memory/`　　🎯 記憶模式：結構化事實與偏好"
                )
                with gr.Row():
                    refresh_mem_btn = gr.Button("🔄 重新整理記憶列表", variant="primary")
                    mem_key_to_delete = gr.Textbox(label="輸入要刪除的 Session Key", scale=3)
                    delete_mem_btn = gr.Button("🗑️ 刪除此記憶", variant="stop")

                mem_status = gr.Markdown()
                mem_table = gr.Dataframe(
                    headers=["Session Key", "輪數", "摘要長度", "最後更新", "摘要預覽"],
                    label="所有 Thread 記憶",
                    interactive=False,
                    wrap=True,
                )
                mem_detail = gr.TextArea(label="📖 摘要完整內容（點選後貼上 Session Key 查詢）", lines=10, interactive=False)
                mem_key_inspect = gr.Textbox(label="查詢 Session Key（輸入後按 Enter）")

            # --- TAB 4: Skill Editor ---
            with gr.Tab("🛠️ Skill Editor"):
                with gr.Row():
                    with gr.Column(scale=1):
                        skill_list = gr.Dropdown(label="Select Skill", choices=[s.name for s in handler.skill_repo.get_all_skills()])
                        refresh_skills_btn = gr.Button("🔄 Refresh List")
                        skill_file_list = gr.Dropdown(label="Select File")
                        save_skill_btn = gr.Button("💾 Save File", variant="primary")
                    with gr.Column(scale=3):
                        skill_editor = gr.Code(label="File Editor", language="markdown", lines=25)

        # --- Events ---
        # apply_preset_btn.click(apply_preset, inputs=[session_key_preset], outputs=[session_key_input])
        
        refresh_models_btn.click(handler.refresh_models, inputs=[provider_radio, api_url_input], outputs=[model_dropdown])
        
        # Skill Editor refresh
        refresh_skills_btn.click(handler.refresh_skills_list, outputs=[skill_list])
        
        # --- Execution Logic ---
        execution_inputs = [
            file_upload_input, 
            provider_radio, 
            api_url_input, 
            model_dropdown, 
            image_max_size_slider, 
            max_agent_steps_slider, 
            system_prompt_input, 
            user_prompt_input, 
            chatbot, 
            session_dropdown
        ]
        execution_outputs = [chatbot, log_output, injected_prompt_output, visual_usage_status, file_upload_input, user_prompt_input]

        run_event = run_btn.click(
            handler.run_agent_task,
            inputs=execution_inputs,
            outputs=execution_outputs
        )
        
        # Also trigger on Enter in the user prompt box
        submit_event = user_prompt_input.submit(
            handler.run_agent_task,
            inputs=execution_inputs,
            outputs=execution_outputs
        )

        stop_btn.click(fn=None, cancels=[run_event, submit_event])
        
        def clear_ui():
            return [], "", "", "<div style='color: gray; font-size: 14px;'>對話已清除 (長期記憶保留)，選擇模型後將顯示 Token 使用量...</div>"
        
        clear_btn.click(clear_ui, None, [chatbot, log_output, injected_prompt_output, visual_usage_status])

        # --- Memory Manager Events ---
        refresh_mem_btn.click(handler.list_memories, outputs=mem_table)
        mem_key_inspect.submit(handler.inspect_memory, inputs=mem_key_inspect, outputs=mem_detail)
        
        def delete_and_refresh(key):
            msg = handler.delete_memory(key)
            table = handler.list_memories()
            return msg, table
        
        delete_mem_btn.click(delete_and_refresh, inputs=mem_key_to_delete, outputs=[mem_status, mem_table])
        
        # Auto-load memories on tab render
        demo.load(handler.list_memories, outputs=mem_table)

        # --- Skill Editor Events ---
        skill_list.change(handler.on_skill_select, inputs=[skill_list], outputs=[skill_file_list, skill_editor])
        
        def load_file_content(skill_name, file_name):
            if not skill_name or not file_name:
                return ""
            return handler.skill_repo.get_skill_details(skill_name, file_name) or ""
        
        skill_file_list.change(load_file_content, inputs=[skill_list, skill_file_list], outputs=[skill_editor])
        
        save_status = gr.Textbox(label="Save Status", visible=True)
        save_skill_btn.click(handler.save_skill_content, inputs=[skill_list, skill_file_list, skill_editor], outputs=[save_status])
        
        # --- Token Usage Auto-Update & Session Switching ---
        usage_inputs = [session_dropdown, provider_radio, api_url_input, model_dropdown]
        
        # Consolidate session switching to avoid double-update
        session_dropdown.change(
            handler.on_session_change, 
            inputs=[session_dropdown, provider_radio, api_url_input, model_dropdown], 
            outputs=[chatbot, visual_usage_status]
        )
        
        model_dropdown.change(handler.get_usage_status, inputs=usage_inputs, outputs=visual_usage_status)
        provider_radio.change(handler.get_usage_status, inputs=usage_inputs, outputs=visual_usage_status)
        
        # For buttons, we only update the dropdown value, which then triggers .change() automatically
        add_session_btn.click(handler.on_add_session_simple, outputs=[session_dropdown])
        delete_session_btn.click(handler.on_delete_session_simple, inputs=[session_dropdown], outputs=[session_dropdown, chatbot])

    return demo
