from core.memory import ThreadMemory

def format_memory_for_prompt(mem: ThreadMemory) -> str:
    """
    Skill-side implementation of how to format the core memory data into a system prompt.
    Now includes standardized key listing to prevent redundant naming.
    """
    parts = []
    
    # 1. User Profile & Preferences
    profile_parts = []
    all_keys = []

    if mem.user_profile:
        p_text = ", ".join(f"{k}: {v}" for k, v in mem.user_profile.items())
        profile_parts.append(f"User Profile: {p_text}")
        all_keys.extend([f"Profile: {k}" for k in mem.user_profile.keys()])

    if mem.preferences:
        pref_text = ", ".join(f"{k}: {v}" for k, v in mem.preferences.items())
        profile_parts.append(f"Preferences: {pref_text}")
        all_keys.extend([f"Pref: {k}" for k in mem.preferences.keys()])

    if mem.agent_rules:
        rules_text = "\n".join(f"  - {k}: {v}" for k, v in mem.agent_rules.items())
        profile_parts.append(f"Agent Action Guidelines:\n{rules_text}")
        all_keys.extend([f"Rule: {k}" for k in mem.agent_rules.keys()])

    if profile_parts:
        parts.append("\n\n".join(profile_parts))

    # 2. Build Memory Keys Info (The list AI must check)
    if all_keys:
        keys_info = "\n\n📋 CURRENTLY REGISTERED MEMORY LABELS (Use these for upserting!):\n- " + "\n- ".join(all_keys)
    else:
        keys_info = "\n\n(No memory labels registered yet. You are free to create standard ones like brand_preference, tone, etc.)"

    content = "\n\n".join(parts) if parts else "(No structured memory recorded yet. Use 'memory' skill to save persona facts and preferences.)"

    return (
        "\n\n"
        "================================================================\n"
        "📌 USER PERSONA & PREFERENCES (Known Context)\n"
        "================================================================\n"
        + content
        + keys_info
        + "\n================================================================\n"
    )
