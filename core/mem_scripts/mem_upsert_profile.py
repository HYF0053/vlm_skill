#!/usr/bin/env python3
"""
mem_upsert_profile.py — Update global_profile.json (standalone, no core imports).

Usage:
    python core/mem_scripts/mem_upsert_profile.py <key> "<value>" --type <preference|profile|agent_rules>
"""
import sys
import os
import json
import argparse
import uuid
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROFILE_PATH = os.path.join(PROJECT_ROOT, "data", "memory", "global_profile.json")

TYPE_MAP = {
    "preference":  "preferences",
    "profile":     "user_profile",
    "agent_rules": "agent_rules",
}

EMPTY_PROFILE = {
    "schema_version": 1,
    "user_profile": {},
    "preferences": {},
    "agent_rules": {},
}


def load_profile() -> dict:
    if not os.path.exists(PROFILE_PATH):
        return {**EMPTY_PROFILE}
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {**EMPTY_PROFILE}


def save_profile(profile: dict) -> None:
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    profile["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = PROFILE_PATH + f".{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PROFILE_PATH)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


def main():
    parser = argparse.ArgumentParser(description="Upsert a key/value into global_profile.json")
    parser.add_argument("key",   help="Memory key (e.g. 'response_language', 'tone')")
    parser.add_argument("value", help="Memory value")
    parser.add_argument("--type", "-t",
                        choices=["preference", "profile", "agent_rules"],
                        default="preference",
                        help="Which section to update (default: preference)")
    args = parser.parse_args()

    profile      = load_profile()
    section_name = TYPE_MAP[args.type]
    section      = profile.get(section_name, {})
    old_val      = section.get(args.key)

    section[args.key]        = args.value
    profile[section_name]    = section

    save_profile(profile)

    if old_val is not None:
        print(f"✅ [{args.type}] '{args.key}' updated: '{old_val}' → '{args.value}'")
    else:
        print(f"✅ [{args.type}] '{args.key}' set to: '{args.value}'")


if __name__ == "__main__":
    main()
