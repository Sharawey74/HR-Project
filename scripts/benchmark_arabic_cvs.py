import os
import glob
import sys
from src.backend import HRBackend

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def run_benchmark():
    print("=== Arabic CV Benchmark Suite ===")
    
    # Initialize Backend
    backend = HRBackend()
    backend.initialize()
    
    cv_dir = "data/test_arabic_cvs"
    cv_files = glob.glob(os.path.join(cv_dir, "*.txt"))
    
    if not cv_files:
        print(f"No CV files found in {cv_dir}")
        return

    print(f"Found {len(cv_files)} CVs to test.\n")
    
    results_summary = []

    for cv_path in cv_files:
        filename = os.path.basename(cv_path)
        print(f"Processing {filename}...")
        
        try:
            with open(cv_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Run pipeline
            result = backend.process_match(text, top_k=3)
            
            # Extract metrics
            candidate_name = result.get('candidate_name', 'Unknown')
            skills_count = result.get('extraction_details', {}).get('skills_count', 0)
            top_matches = result.get('top_matches', [])
            
            # Store best match
            best_match = top_matches[0] if top_matches else {}
            best_role = best_match.get('job_title', 'N/A')
            best_score = best_match.get('confidence', 0.0) * 100
            match_quality = best_match.get('decision', 'N/A')
            
            print(f"  -> Name: {candidate_name}")
            print(f"  -> Skills: {skills_count}")
            print(f"  -> Top Match: {best_role} ({best_score:.1f}%) [{match_quality}]")
            print("-" * 30)
            
            results_summary.append({
                "file": filename,
                "skills": skills_count,
                "role": best_role,
                "score": best_score,
                "quality": match_quality
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")

    # Final Summary Table
    print("\n=== Final Benchmark Results ===")
    print(f"{'File':<25} | {'Skills':<6} | {'Top Match':<30} | {'Score':<6} | {'Quality'}")
    print("-" * 85)
    for r in results_summary:
        print(f"{r['file']:<25} | {r['skills']:<6} | {r['role']:<30} | {r['score']:.1f}% | {r['quality']}")

if __name__ == "__main__":
    run_benchmark()
