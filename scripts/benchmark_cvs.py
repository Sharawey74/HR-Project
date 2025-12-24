import sys
import os
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.backend import HRBackend

def run_benchmark():
    print("Initializing HR Backend...")
    backend = HRBackend()
    
    cv_path = Path("data/benchmark_cvs.json")
    if not cv_path.exists():
        print(f"Error: {cv_path} not found.")
        return
        
    with open(cv_path, "r", encoding="utf-8") as f:
        cvs = json.load(f)
        
    print(f"Loaded {len(cvs)} CVs for benchmarking.")
    print("="*60)
    
    total_start = time.time()
    
    for cv in cvs:
        name = cv['name']
        role = cv['role']
        text = cv['text']
        
        print(f"\nProcessing: {name} ({role})...")
        start = time.time()
        
        result = backend.process_match(profile_text=text, top_k=3)
        
        duration = time.time() - start
        
        matches = result.get('top_matches', [])
        
        print(f"  > Time: {duration:.2f}s | Matches Found: {len(matches)}")
        
        for i, match in enumerate(matches, 1):
            title = match.get('job_title', 'Unknown Title')
            score = match.get('final_score', 0)
            decision = match.get('decision', 'N/A')
            print(f"    {i}. [{decision}] {title} ({score:.1%})")
            if i == 1:
                print(f"       Explanation: {match.get('explanation', 'No explanation')}")
            
    total_duration = time.time() - total_start
    print("="*60)
    print(f"Benchmark Complete in {total_duration:.2f}s")

if __name__ == "__main__":
    # Ensure env var is set to load all jobs
    os.environ['MAX_JOBS'] = '0'
    run_benchmark()
