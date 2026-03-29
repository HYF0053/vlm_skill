#!/usr/bin/env python3
"""
mem_read_profile.py — Read and print global_profile.json (standalone, no core imports).

Usage:
    python core/mem_scripts/mem_read_profile.py
    python core/mem_scripts/mem_read_profile.py --type preference
"""
import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROFILE_PATH = os.path.join(PROJECT_ROOT, "data", "memory", "global_profile.json")

TYPE_MAP = {
    "preference":  "preferences",
    "profile":     "user_profile",
    "agent_rules": "agent_rules",
}


def load_profile() -> dict:
    if not os.path.exists(PROFILE_PATH):
        return {"schema_version": 1, "user_profile": {}, "preferences": {}, "agent_rules": {}}
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Could not read profile: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Read global_profile.json")
    parser.add_argument("--type", "-t",
                        choices=["preference", "profile", "agent_rules"],
                        default=None,
                        help="Filter to a specific section (default: all)")
    args = parser.parse_args()

    profile = load_profile()

    if args.type:
        section_name = TYPE_MAP[args.type]
        data = profile.get(section_name, {})
        print(f"\n📋 [{args.type}]")
        if data:
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print("  (empty)")
    else:
        print(f"\n📋 Global Profile  (last updated: {profile.get('last_updated_at', 'unknown')})")
        for section_key, section_label in [
            ("user_profile", "User Profile"),
            ("preferences",  "Preferences"),
            ("agent_rules",  "Agent Rules"),
        ]:
            data = profile.get(section_key, {})
            print(f"\n  🔹 {section_label}:")
            if data:
                for k, v in data.items():
                    print(f"    {k}: {v}")
            else:
                print("    (empty)")


if __name__ == "__main__":
    main()
