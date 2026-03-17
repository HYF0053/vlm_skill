import sys
import os
import argparse

# Add project root to path for imports
sys.path.append("/home/ubuntu/ai-agent-platform")
sys.path.append("/home/ubuntu/vlm_skill")

from core.memory import MemoryStore

def main():
    parser = argparse.ArgumentParser(description="Upsert short-term JSON memory (Preferences, Profile, Project).")
    parser.add_argument("key", help="The memory key (e.g., 'os', 'code_style')")
    parser.add_argument("value", help="The memory value content")
    parser.add_argument("--type", choices=["preference", "profile", "project", "fact"], default="preference", 
                        help="Memory type (Note: 'fact' is recommended for RAG instead)")
    
    args = parser.parse_args()
    
    session_key = os.environ.get("SESSION_ID", "main")
    
    store = MemoryStore()
    msg = store.upsert_memory(session_key, args.key, args.value, args.type)
    print(msg)

if __name__ == "__main__":
    main()
