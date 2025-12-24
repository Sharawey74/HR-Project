
import sys
import importlib.util

print("=== AutoGen Debug Check ===")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

# Check sys.path
# print("Sys.path:", sys.path)

# Check imports
packages = ['autogen', 'pyautogen', 'openai', 'flaml']
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    print(f"Package '{pkg}': {'FOUND' if spec else 'MISSING'}")
    if spec:
        print(f"  -> Path: {spec.origin}")

try:
    from autogen import ConversableAgent
    print("SUCCESS: from autogen import ConversableAgent worked.")
except ImportError as e:
    print(f"FAILURE importing ConversableAgent: {e}")
except Exception as e:
    print(f"ERROR: {e}")
