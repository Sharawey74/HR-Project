
import sys
import os
sys.path.insert(0, os.getcwd())

from src.backend import HRBackend

def verify_golden():
    print("=== Golden Arabic CV Verification ===")
    backend = HRBackend()
    backend.initialize()
    
    cv_path = "data/test_arabic_cvs/golden_arabic_cv.txt"
    if not os.path.exists(cv_path):
        print(f"Error: CV file not found at {cv_path}")
        return

    with open(cv_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process
    results = backend.process_match(text, top_k=3)
    
    # Output Details
    print(f"\nCandidate: {results.get('candidate_name')}")
    print(f"Language: {results.get('language')}")
    
    print("\n--- Top Matches ---")
    matches = results.get('top_matches', [])
    for i, m in enumerate(matches, 1):
        print(f"#{i} {m.get('job_title')} (Job ID: {m.get('job_id')})")
        print(f"   Confidence: {m.get('confidence')*100:.1f}%")
        print(f"   Decision: {m.get('decision')}")
        print(f"   Skills Matched: {len(m.get('skill_match', {}).get('matched_skills', []))}")
        print("-" * 20)

    # Check against AI Engineer (Job 21)
    best = matches[0] if matches else {}
    if "AI Engineer" in best.get('job_title', '') and best.get('confidence', 0) > 0.8:
        print("\nSUCCESS: Golden CV achieved high confidence match for AI Engineer!")
    else:
        print("\nWARNING: Match target missed or low score.")

if __name__ == "__main__":
    verify_golden()
