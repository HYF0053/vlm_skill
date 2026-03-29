# core/mem_scripts — Standalone memory management scripts.
# These scripts are self-contained: they do NOT import from core/.
# They only use: stdlib, requests, config/memo.json, and direct file I/O.
# This makes them safe to call via run_cli_command() without any import side-effects.
