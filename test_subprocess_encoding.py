import subprocess
import sys

# Simulate the agent's execute_script or run_cli_command
pdf_path = r"C:\Users\iaian\Downloads\2403.14403v2.pdf"
cmd = f"pdftotext \"{pdf_path}\" -" # Output to stdout

print(f"Running command: {cmd}")
try:
    # This is what's currently in core/skills.py
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    print(f"Return code: {result.returncode}")
    print(f"Stderr: {result.stderr}")
    print(f"Stdout length: {len(result.stdout)}")
    print(result.stdout[:200])
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError caught: {e}")
except Exception as e:
    print(f"Other error: {e}")

print("\n--- Testing with explicit utf-8 encoding and errors='replace' ---")
try:
    result = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    print(f"Return code: {result.returncode}")
    print(f"Stderr: {result.stderr}")
    print(f"Stdout length: {len(result.stdout)}")
    print(result.stdout[:200])
except Exception as e:
    print(f"Error: {e}")
