import subprocess
import sys

# Script that outputs UTF-8 but calling process expects CP950
code = """
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
print('Hello \u4f60\u597d') # Hello Nǐ hǎo (Chinese)
"""

with open("temp_script.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Trying to capture UTF-8 output with default text=True (Windows CP950)...")
try:
    # This simulates the PREVIOUS state of core/skills.py on Windows
    result = subprocess.run([sys.executable, "temp_script.py"], capture_output=True, text=True, timeout=10)
    print("Success (unexpected on some Windows systems!):")
    print(result.stdout)
except UnicodeDecodeError as e:
    print(f"Caught expected UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Other error: {e}")

print("\nTrying with explicit encoding='utf-8'...")
try:
    result = subprocess.run([sys.executable, "temp_script.py"], capture_output=True, encoding="utf-8", timeout=10)
    print("Success with encoding='utf-8':")
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")
