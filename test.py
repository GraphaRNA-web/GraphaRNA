
import subprocess
import os
import json

def get_rna_lddt(model_path, ref_path):
    abs_model = os.path.abspath(model_path)
    abs_ref = os.path.abspath(ref_path)
    current_dir = os.path.abspath(".") 
    report_path = os.path.join(current_dir, "ost_report.json")
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_model}:/model.pdb",
        "-v", f"{abs_ref}:/ref.pdb",
        "-v", f"{current_dir}:/outdir",  
        "registry.scicore.unibas.ch/schwede/openstructure:latest",
        "compare-structures",
        "-m", "/model.pdb",
        "-r", "/ref.pdb",
        "--lddt",
        "--local-lddt",
        # --- THE FIX IS HERE ---
        "--chain-mapping", "B:A",  # Force Model Chain A to map to Ref Chain B
        "--lddt-no-stereochecks",
        # Notice we removed the "-rna" flag so it aligns by sequence, not number
        # -----------------------
        "-o", "/outdir/ost_report.json"
    ]
    
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                data = json.load(f)
            
            os.remove(report_path)
            
            global_score = data.get("lddt", 0.0)
            raw_local_scores = data.get("local_lddt", {})
            
            clean_local_scores = {}
            for res_id, score in raw_local_scores.items():
                clean_id = res_id.rstrip('.')
                clean_local_scores[clean_id] = 0.0 if score is None else score
                
            return global_score, clean_local_scores
            
        return 0.0, {}
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to run OpenStructure:\n{e.output.decode()}")
        return 0.0, {}

# Test it with your two different files
model_file = "samples/3_ar.dotseq.pdb"
ref_file = "5T5H_1_B_B_873_G.pdb"

global_score, local_scores = get_rna_lddt(model_file, ref_file)

print(f"Global Score: {global_score}")
print("\nClean Per-Residue Scores:")
for res, score in local_scores.items():
    print(f"{res}: {score}")