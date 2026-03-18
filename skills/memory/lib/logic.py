from core.memory import ThreadMemory

def format_memory_for_prompt(mem: ThreadMemory) -> str:
    """
    Skill-side implementation of how to format the core memory data into a system prompt.
    """
    parts = []
    
    # 1. User Profile & Preferences
    profile_parts = []
    if mem.user_profile:
        p_text = ", ".join(f"{k}: {v}" for k, v in mem.user_profile.items())
        profile_parts.append(f"User Profile: {p_text}")
    if mem.preferences:
        pref_text = ", ".join(f"{k}: {v}" for k, v in mem.preferences.items())
        profile_parts.append(f"Preferences: {pref_text}")
    if mem.agent_rules:
        rules_text = "\n".join(f"  - {k}: {v}" for k, v in mem.agent_rules.items())
        profile_parts.append(f"Agent Action Guidelines:\n{rules_text}")
    if profile_parts:
        parts.append("\n\n".join(profile_parts))

    content = "\n\n".join(parts) if parts else "(No structured memory recorded yet. Use 'memory' skill to save persona facts and preferences.)"

    return (
        "\n\n"
        "================================================================\n"
        "📌 USER PERSONA & PREFERENCES (Known Context)\n"
        "================================================================\n"
        + content
        + "\n================================================================\n"
    )
