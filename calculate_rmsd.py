import sys
import warnings
from Bio.PDB import PDBParser, Superimposer
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

def calc_rmsd(pdb1, pdb2):
    parser = PDBParser()
    try:
        s1 = parser.get_structure('1', pdb1)
        s2 = parser.get_structure('2', pdb2)
        
        # Ekstrakcja atomów z obu struktur
        atoms1 = list(s1.get_atoms())
        atoms2 = list(s2.get_atoms())
        
        if len(atoms1) != len(atoms2):
            return -1.0 # Błąd - różna liczba atomów
            
        sup = Superimposer()
        sup.set_atoms(atoms1, atoms2)
        return sup.rms
    except Exception as e:
        return -1.0

if __name__ == "__main__":
    pdb_orig = sys.argv[1]
    pdb_impr = sys.argv[2]
    rmsd = calc_rmsd(pdb_orig, pdb_impr)
    print(f"{rmsd:.4f}")
