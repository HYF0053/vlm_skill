---
name: memory
description: Structured long-term memory management for remembering user profiles, project status, and important facts.
---

# Memory Skill

This skill provides the agent with structured long-term memory. Instead of relying on automatic summarization, the agent explicitly decides what to remember using the `upsert_memory` tool.

## Concepts

1. **Facts**: Specific, discrete pieces of information. These are added to a growing list.
2. **Preferences**: Key-value pairs that define how the user wants things done. Overwrites previous values for the same key.
3. **User Profile**: Key-value pairs defining the user's background (e.g., OS, role, skills).
4. **Project Status**: Key-value pairs defining the current state of the project.

## How to use

When you encounter information that is likely to be useful in future turns or sessions, use the `upsert_memory` tool.

### Examples:

- `upsert_memory(key="os", value="Ubuntu 24.04", mem_type="profile")`
- `upsert_memory(key="editor", value="VS Code", mem_type="preference")`
- `upsert_memory(key="database_path", value="./data/prod.db", mem_type="project")`
- `upsert_memory(key="user_birthday", value="June 15", mem_type="fact")`

## Currently Available Tools

- `upsert_memory(key: str, value: str, mem_type: str = 'fact') -> str`
    - `key`: The name of the property.
    - `value`: The information to store.
    - `mem_type`: One of `fact`, `preference`, `profile`, `project`.
