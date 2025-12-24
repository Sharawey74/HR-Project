import json
import os
from src.agents.agent1_parser import RawParser
from src.agents.agent2 import CandidateExtractor
from src.agents.agent3 import JobMatcher
from src.backend import HRBackend

def run_debug_robotics():
    print("=== Robotics CV Debug Target ===")
    backend = HRBackend()
    backend.initialize()
    
    cv_path = "data/test_arabic_cvs/cv10_robotics_expert.txt"
    
    with open(cv_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Run full pipeline with details
    results = backend.process_match(text, top_k=5)
    
    print(f"\nResults Keys: {list(results.keys())}")
    
    # Check Agent 1 output (Translation/Segmentation)
    candidate_profile = results.get('extraction_details', {})
    print(f"\n--- Extraction Details ---")
    print(f"Candidate Name: {results.get('candidate_name')}")
    print(f"Language: {results.get('language')}")
    print(f"Is Bilingual: {results.get('is_bilingual')}")
    print(f"Skills Extracted Count: {candidate_profile.get('skills_count', 0)}")
    print(f"Experience Years: {candidate_profile.get('experience_years')}")
    
    top_matches = results.get('top_matches', [])
    
    print("\n--- Top Matches ---")
    for i, m in enumerate(top_matches):
        confidence = m.get('confidence', 0)
        print(f"[{i+1}] {m.get('job_title', 'No Title')} | Confidence: {confidence*100:.1f}%")
        print(f"    Match Quality: {m.get('decision', 'N/A')}")
        skill_match = m.get('skill_match', {})
        print(f"    Matched Skills: {skill_match.get('matched_skills', [])}")
        print(f"    Reason: {m.get('explanation', '')[:150]}...")
        print("-" * 20)
        print("-" * 20)

if __name__ == "__main__":
    run_debug_robotics()
