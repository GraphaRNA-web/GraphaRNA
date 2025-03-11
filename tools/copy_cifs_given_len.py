import os
import shutil
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure
from tqdm import tqdm


data_dir = '/home/mjustyna/data/full_PDB/'
output_dir = '/home/mjustyna/data/full_PDB-500/'

files = os.listdir(data_dir)
files = [f for f in files if f.endswith('.pdb')]
os.makedirs(output_dir, exist_ok=True)

for f in tqdm(files):
    # if file size is bigger than 1MB, skip it
    if os.path.getsize(os.path.join(data_dir, f)) > 1000000:
        continue
    if os.path.exists(os.path.join(output_dir, f)):
        continue
    try:
        with open(os.path.join(data_dir, f), 'r') as file:
            structure3d = read_3d_structure(file, 1)
            if len(structure3d.residues) <= 500:
                shutil.copy(os.path.join(data_dir, f), os.path.join(output_dir, f))

    except IndexError:
        print(f"Error in file {f}")
        continue