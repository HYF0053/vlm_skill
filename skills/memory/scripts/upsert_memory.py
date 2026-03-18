import sys
import os
import argparse

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.memory import MemoryStore

def upsert_memory(store: MemoryStore, session_key: str, key: str, value: str, mem_type: str = "preference") -> str:
    """
    Update structured memory in the store.
    """
    key = key.strip()
    msg = ""

    with store._lock_session(session_key) as mem:
        if mem_type == "preference":
            old_val = mem.preferences.get(key)
            mem.preferences[key] = value
            msg = f"Preference updated: {key} = {value}"
            if old_val:
                msg += f" (previous value '{old_val}' was overwritten. If you wanted to keep it, you should have merged it in the 'value'.)"
        elif mem_type == "profile":
            old_val = mem.user_profile.get(key)
            mem.user_profile[key] = value
            msg = f"User profile updated: {key} = {value}"
            if old_val:
                msg += f" (previous value '{old_val}' was overwritten)"
        elif mem_type == "project":
            old_val = mem.current_project_status.get(key)
            mem.current_project_status[key] = value
            msg = f"Project status updated: {key} = {value}"
            if old_val:
                msg += f" (previous value '{old_val}' was overwritten)"
        else:
            return f"Unknown memory type: {mem_type}"

    return msg

def main():
    parser = argparse.ArgumentParser(description="Upsert short-term JSON memory (Preferences, Profile, Project).")
    parser.add_argument("key", help="The memory key (e.g., 'os', 'code_style')")
    parser.add_argument("value", help="The memory value content")
    parser.add_argument("--type", choices=["preference", "profile", "project"], default="preference", 
                        help="Memory type (preference, profile, project)")
    
    args = parser.parse_args()
    
    session_key = os.environ.get("SESSION_ID", "main")
    
    store = MemoryStore()
    msg = upsert_memory(store, session_key, args.key, args.value, args.type)
    print(msg)

if __name__ == "__main__":
    main()
