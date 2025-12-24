import sys
import traceback

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Testing python-docx ---")
try:
    import docx
    print(f"✅ python-docx imported successfully. Version: {docx.__version__ if hasattr(docx, '__version__') else 'unknown'}")
except ImportError:
    print("❌ python-docx FAILED to import.")
    traceback.print_exc()
except Exception as e:
    print(f"❌ python-docx error: {e}")
    traceback.print_exc()

print("\n--- Testing autogen ---")
try:
    import autogen
    print(f"✅ autogen imported successfully. Version: {autogen.__version__ if hasattr(autogen, '__version__') else 'unknown'}")
except ImportError:
    print("❌ autogen FAILED to import.")
    traceback.print_exc()
except Exception as e:
    print(f"❌ autogen error: {e}")
    traceback.print_exc()
