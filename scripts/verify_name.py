
import sys
import os
sys.path.insert(0, os.getcwd())

from src.agents.agent1_parser import RawParser
from src.agents.agent2 import CandidateExtractor

def verify():
    print("=== Name Verification Script ===")
    p = RawParser()
    e = CandidateExtractor()
    
    cv_path = "data/test_arabic_cvs/cv10_robotics_expert.txt"
    
    # 1. Parse
    print(f"Parsing {cv_path}...")
    parsed = p.parse_profile_with_language_detection(open(cv_path, 'r', encoding='utf-8').read())
    print(f"Language: {parsed.get('language')}")
    print(f"Original Text Sample: {parsed.get('original_text')[:50]}...")
    
    # 2. Extract
    print("Extracting...")
    text_to_process = parsed.get('translated_text') or parsed.get('original_text')
    print(f"Processing Text Sample: {text_to_process[:50]}...")
    
    profile = e.extract(text_to_process)
    name = profile.get('name')
    print(f"Extracted Name: {name}")
    
    # Check
    # "د. فاطمة سعيد الكعبي" is the expected logical order
    expected = "د. فاطمة سعيد الكعبي"
    if name and "فاطمة" in name:
        print("SUCCESS: Name extracted correclty.")
    else:
        print("FAILURE: Name mismatch or unreadable.")

if __name__ == "__main__":
    verify()
