from grapharna.utils.calculate_lddt import calculate_lddt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--prediction', type=str, default=None, help='Prediction file')
parser.add_argument('--reference', type=str, default=None, help='Reference file')

args = parser.parse_args()
print("--- Calculating lDDT Scores ---")


global_score, local_scores = calculate_lddt(args.prediction, args.reference)

print(f"\nGlobal Score: {global_score}")
print("\nClean Per-Residue Scores:")
for res, score in local_scores.items():
    print(f"{res}: {score}")