#!/bin/bash

echo "=== Automatyczny benchmark solverów (Fast Selection + Runtime Validation + DEBUG) ==="

# 1. Konfiguracja i aktywacja środowiska
source venv/bin/activate 2>/dev/null || source ../venv3.10/bin/activate

RESULTS_DIR="results_benchmark"
METADATA="${RESULTS_DIR}/benchmark_rmsd_report.txt"
REF_EXP_DIR="/home/pwlklr/Studia/GraphaRNA-CD/RNA-GNN-test-pdb"

# --- ZMIENNE KONTROLNE ---
TARGET_1_SEG=30
TARGET_2_SEG=25
TARGET_3_SEG=25

MAREK_DIR="/home/pwlklr/Studia/GraphaRNA-CD/grapharna-eval-seed=0"
MAREK_1="${MAREK_DIR}/1_segments"
MAREK_2="${MAREK_DIR}/2_segments"
MAREK_3="${MAREK_DIR}/3_segments"
# ---------------------------------

mkdir -p "$RESULTS_DIR"

# ===============================================================================
# APLIKACJA WALIDATORA (Poprawiony parser wejściowy)
# ===============================================================================
cat << 'EOF' > rna_validator_standalone.py
import sys
import os
import traceback
from collections import deque
import warnings
warnings.filterwarnings("ignore")
from grapharna.preprocess_rna_pdb import get_dotseq_from_pdb

VALID_BRACKETS = ".()[]{}<>AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"
VALID_NUCLEOTIDES = "ACGU"
VALID_PAIRS = "AUCGCGAUUGGUUAAA"
MAX_RNA_LENGTH = 10000
SEPARATOR_CHOICES = ["&", "+"]

class RnaValidator:
    def __init__(self, fasta_raw: str) -> None:
        self.fasta_raw: str = fasta_raw
        self.validBrackets: str = VALID_BRACKETS + " "
        self.validNucleotides: set[str] = set(VALID_NUCLEOTIDES + " ")
        self.validPairs: list[str] = [VALID_PAIRS[i : i + 2] for i in range(0, len(VALID_PAIRS), 2)]
        self.parsingResult: bool = True
        self.errorList: list[str] = []
        self.strandSeparator: str | None = None
        self.FastaFileParse()
        print(self.fasta_raw)

    def FastaFileParse(self) -> None:
        nucleotides: str = ""
        dotBracket: str = ""
        inputStructureSplit: list[str] = [item.strip() for item in self.fasta_raw.split("\n") if (item.strip() != "" and item[0] != "#")]
        potentialNameLines: list[str] = inputStructureSplit[::3]
        areNameLines: list[bool] = [i[0] == ">" for i in potentialNameLines]

        if all(areNameLines):
            containsStrandNames = True
        elif any(areNameLines):
            self.parsingResult = False
            self.errorList.append("Parsing error: Inconsistent strand naming")
            return None
        else:
            containsStrandNames = False

        if containsStrandNames:
            step = 3; rna_index = 1; dot_index = 2
        else:
            step = 2; rna_index = 0; dot_index = 1

        for i in range(0, len(inputStructureSplit), step):
            currentStrand: list[str] = inputStructureSplit[i : i + step]
            if len(currentStrand) < step:
                self.parsingResult = False
                self.errorList.append("Parsing error: Missing Lines")
                return None
            if not any(i in self.validBrackets for i in currentStrand[dot_index]) and all(i in self.validNucleotides for i in currentStrand[dot_index].upper()):
                self.parsingResult = False
                self.errorList.append("Parsing error: Wrong line order")
                return None
            
            if self.strandSeparator is None:
                for separator in SEPARATOR_CHOICES:
                    if currentStrand[rna_index].find(separator) != -1 and self.strandSeparator is None:
                        self.strandSeparator = separator
                    elif currentStrand[rna_index].find(separator) != -1 and self.strandSeparator is not None:
                        self.parsingResult = False
                        self.errorList.append("Parsing error: Mismatching strand separators")
            if self.strandSeparator is None:
                self.strandSeparator = "N"

            if currentStrand[rna_index].count(self.strandSeparator) != currentStrand[dot_index].count(self.strandSeparator):
                self.errorList.append("Parsing error: Mismatching strand separators")
                self.parsingResult = False
                return None
            
            if any([len(pair[0]) != len(pair[1]) for pair in zip(currentStrand[rna_index].split(self.strandSeparator), currentStrand[dot_index].split(self.strandSeparator))]):
                self.parsingResult = False
                self.errorList.append("Parsing error: Mismatching strand lengths")
                return None

            nucleotides += currentStrand[rna_index] + " "
            dotBracket += currentStrand[dot_index] + " "

        if self.strandSeparator is not None and self.strandSeparator != "N":
            sep = self.strandSeparator
            self.parsedStructure: str = nucleotides.strip().upper().replace("T", "U").replace(sep, " ") + "\n" + dotBracket.strip().replace(sep, " ")
        else:
            self.parsedStructure = nucleotides.strip().upper().replace("T", "U") + "\n" + dotBracket.strip()

    def ValidateRna(self) -> dict:
        validatedRna = ""
        validationResult = False
        fixSuggested = False
        mismatchingBrackets = []
        incorrectPairs = []

        if not self.parsingResult:
            return {"Validation Result": False, "Error List": self.errorList}

        inputStr = self.parsedStructure
        rnaSplit = inputStr.split("\n")
        rna = rnaSplit[0]
        dotBracket = rnaSplit[1]

        if len(rna) == 0 or len(rna.replace(" ", "")) > MAX_RNA_LENGTH:
            self.errorList.append("Invalid length")
            return {"Validation Result": False, "Error List": self.errorList}
        if len(rna) != len(dotBracket):
            self.errorList.append(f"Length mismatch: RNA({len(rna)}) vs DotBracket({len(dotBracket)})")
            return {"Validation Result": False, "Error List": self.errorList}

        invalidCharacters = set(char for char in rna if char not in self.validNucleotides)
        if len(invalidCharacters) > 0:
            self.errorList.append(f"Invalid characters in RNA: {invalidCharacters}")
            return {"Validation Result": False, "Error List": self.errorList}

        invalidBrackets = set(char for char in dotBracket if char not in self.validBrackets)
        if len(invalidBrackets) > 0:
            self.errorList.append(f"Invalid brackets in DotBracket: {invalidBrackets}")
            return {"Validation Result": False, "Error List": self.errorList}

        bracketStacks, suggestedDotBracketFixList, mismatchingBrackets, incorrectPairs, allPairs = self.stackCheck(dotBracket, rna)

        for stack in bracketStacks.values():
            for bracket in stack:
                mismatchingBrackets.append(bracket)
                suggestedDotBracketFixList[bracket] = "."
                
        if "".join(suggestedDotBracketFixList) != dotBracket:
            validationResult = True
            fixSuggested = True
            validatedRna = rna + "\n" + "".join(suggestedDotBracketFixList)
        else:
            validationResult = True
            validatedRna = self.parsedStructure

        return {"Validation Result": validationResult, "Validated RNA": validatedRna}

    def stackCheck(self, dotBracket: str, rna: str):
        bracketStacks = {self.validBrackets[i : i + 2]: deque() for i in range(0, len(self.validBrackets), 2) if self.validBrackets[i] != "."}
        allPairs, mismatchingBrackets, incorrectPairs = [], [], []
        openingLookup = {pair[0]: pair for pair in bracketStacks.keys()}
        closingLookup = {pair[1]: pair for pair in bracketStacks.keys()}
        suggestedDotBracketFixList = list(dotBracket)

        for i in range(len(dotBracket)):
            if dotBracket[i] in openingLookup:
                bracketStacks[openingLookup[dotBracket[i]]].append(i)
            elif dotBracket[i] in closingLookup:
                if len(bracketStacks[closingLookup[dotBracket[i]]]) > 0:
                    index = bracketStacks[closingLookup[dotBracket[i]]][-1]
                    if rna[index] + rna[i] in self.validPairs:
                        allPairs.append((index, i))
                        bracketStacks[closingLookup[dotBracket[i]]].pop()
                    else:
                        incorrectPairs.append((index, i))
                        suggestedDotBracketFixList[i] = "."
                        suggestedDotBracketFixList[index] = "."
                        bracketStacks[closingLookup[dotBracket[i]]].pop()
                else:
                    mismatchingBrackets.append(i)
                    suggestedDotBracketFixList[i] = "."
        return bracketStacks, suggestedDotBracketFixList, mismatchingBrackets, incorrectPairs, allPairs

if __name__ == "__main__":
    pdb_id = sys.argv[1]
    pdb_path = sys.argv[2]
    out_path = sys.argv[3]

    try:
        res = get_dotseq_from_pdb(pdb_path)
        if isinstance(res, tuple) and len(res) >= 2:
            fasta_raw = f">{pdb_id}\n{res[0]}\n{res[1]}\n"
        elif isinstance(res, str):
            # POPRAWKA: Rozpoznanie poprawnie wygenerowanego tekstu z tagami >strand
            if res.lstrip().startswith(">"):
                fasta_raw = res.strip()
            else:
                fasta_raw = f">{pdb_id}\n{'A'*len(res.strip())}\n{res.strip()}\n"
        else:
            print("Błąd: funkcja get_dotseq_from_pdb nie zwróciła poprawnego wyniku", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Błąd parsera natywnego get_dotseq_from_pdb: {str(e)}", file=sys.stderr)
        sys.exit(1)

    try:
        validator = RnaValidator(fasta_raw)
        result = validator.ValidateRna()
    except Exception as e:
        print(f"Wyjątek wewnątrz RnaValidator: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if result.get("Validation Result"):
        # Gwarancja, że na wyjściu pojawi się nazwa PDB wymagana przez GraphaRNA
        final_str = f">{pdb_id}\n" + result["Validated RNA"].strip() + "\n"
        with open(out_path, "w") as f:
            f.write(final_str)
        sys.exit(0)
    else:
        print(f"Błędy Walidatora: {result.get('Error List')}", file=sys.stderr)
        sys.exit(1)
EOF
# ===============================================================================

echo "Raport z testów (Wszystkie metryki liczone względem Natywnego Kryształu Eksperymentalnego)" > "$METADATA"
echo "Data: $(date)" >> "$METADATA"
echo "=======================================================================================================================" >> "$METADATA"
printf "%-25s | %-3s | %-8s | %-6s | %-10s | %-15s | %-15s | %-10s\n" "Struktura" "Seg" "Solver" "Kroki" "Czas [s]" "RMSD (do Nat)" "lDDT (do Nat)" "INF (do Nat)" >> "$METADATA"
printf "%-25s | %-3s | %-8s | %-6s | %-10s | %-15s | %-15s | %-10s\n" "-------------------------" "---" "--------" "------" "----------" "---------------" "---------------" "----------" >> "$METADATA"

echo "[*] Błyskawiczna inicjalizacja doboru próbek z wykorzystaniem indeksów..."
SELECTED_ITEMS=()
COUNT_1=0; COUNT_2=0; COUNT_3=0

echo " -> Faza 1: Przeszukiwanie własnego Cache'u w folderze results_benchmark"
for file in $(find "$RESULTS_DIR" -maxdepth 1 -type f -name "*_ddpm_5000_baseline_AA.pdb" 2>/dev/null | shuf); do
    base=$(basename "$file" "_ddpm_5000_baseline_AA.pdb")
    if [ ! -f "$REF_EXP_DIR/${base}.pdb" ]; then continue; fi
    
    if [ -f "$MAREK_1/${base}_AA.pdb" ] && [ $COUNT_1 -lt $TARGET_1_SEG ]; then
        SELECTED_ITEMS+=("${base}:1"); ((COUNT_1++))
        echo "   [Cache] + $base (1-seg) | Stan: $COUNT_1/$TARGET_1_SEG"
    elif [ -f "$MAREK_2/${base}_AA.pdb" ] && [ $COUNT_2 -lt $TARGET_2_SEG ]; then
        SELECTED_ITEMS+=("${base}:2"); ((COUNT_2++))
        echo "   [Cache] + $base (2-seg) | Stan: $COUNT_2/$TARGET_2_SEG"
    elif [ -f "$MAREK_3/${base}_AA.pdb" ] && [ $COUNT_3 -lt $TARGET_3_SEG ]; then
        SELECTED_ITEMS+=("${base}:3"); ((COUNT_3++))
        echo "   [Cache] + $base (3-seg) | Stan: $COUNT_3/$TARGET_3_SEG"
    fi
done

echo " -> Faza 2: Błyskawiczne dopełnianie brakujących struktur (tylko na podstawie nazwy)"
fill_gap() {
    local source_dir=$1
    local target_count=$2
    local current_count=$3
    local seg_val=$4

    if [ "$current_count" -lt "$target_count" ]; then
        if [ ! -d "$source_dir" ]; then return; fi
        
        for file in $(find "$source_dir" -type f -name "*_AA.pdb" 2>/dev/null | shuf); do
            if [ "$current_count" -ge "$target_count" ]; then break; fi
            base=$(basename "$file" "_AA.pdb")
            
            if echo "${SELECTED_ITEMS[@]}" | grep -qw "$base"; then continue; fi
            
            if [ -f "$REF_EXP_DIR/${base}.pdb" ]; then
                SELECTED_ITEMS+=("${base}:${seg_val}"); ((current_count++))
                echo "   [Indeks]+ $base (${seg_val}-seg) | Stan: $current_count/$target_count"
            fi
        done
    fi
}

fill_gap "$MAREK_1" $TARGET_1_SEG $COUNT_1 1
fill_gap "$MAREK_2" $TARGET_2_SEG $COUNT_2 2
fill_gap "$MAREK_3" $TARGET_3_SEG $COUNT_3 3

echo "========================================================"
echo "Gotowa pula struktur testowych: ${#SELECTED_ITEMS[@]}"
echo "========================================================"

run_test() {
    local pdb_id=$1
    local segments=$2
    local sampler=$3
    local steps=$4
    local skip_type=$5
    local native_exp_pdb_path=$6   
    
    local dotseq_file="${RESULTS_DIR}/${pdb_id}.dotseq"
    local cg_out_name="${pdb_id}_${sampler}_${steps}_${skip_type}_CG.pdb"
    local cg_out_path="${RESULTS_DIR}/${cg_out_name}"
    local aa_out_path="${RESULTS_DIR}/${pdb_id}_${sampler}_${steps}_${skip_type}_AA.pdb"

    echo "========================================================"
    echo "    -> URUCHAMIANIE: Solver: $sampler | Kroki: $steps"
    echo "========================================================"
    
    rm -f "$cg_out_path"
    rm -f "$aa_out_path"

    START_TIME=$(date +%s)

    grapharna --input="$dotseq_file" --output-folder="$RESULTS_DIR" --output-name="$cg_out_name" \
              --seed=0 --timesteps=5000 --sampler="$sampler" --steps="$steps"

    if [ -f "$cg_out_path" ]; then
        echo "    Uruchamianie Arena (CG -> All-Atom)..."
        ./Arena/Arena "$cg_out_path" "$aa_out_path" 5
    fi
    
    END_TIME=$(date +%s)
    EXEC_TIME="$((END_TIME - START_TIME)) s"

    if [ -f "$aa_out_path" ]; then
        if [ -f "$native_exp_pdb_path" ]; then
            RAW_RMSD_OUTPUT=$(python3 calculate_rmsd.py "$native_exp_pdb_path" "$aa_out_path" 2>&1)
            RMSD_VAL=$(echo "$RAW_RMSD_OUTPUT" | tail -n 1 | awk '{print $NF}')
            if ! [[ "$RMSD_VAL" =~ ^[0-9]+([.][0-9]+)?$ ]]; then RMSD_VAL="Błąd_RMSD"; fi

            RAW_LDDT_OUTPUT=$(python3 compare.py --prediction "$aa_out_path" --reference "$native_exp_pdb_path" 2>&1)
            LDDT_VAL=$(echo "$RAW_LDDT_OUTPUT" | grep "Global Score:" | awk '{print $3}')
            if [ -z "$LDDT_VAL" ] || [[ "$LDDT_VAL" == "0.0" ]]; then LDDT_VAL="Błąd_lDDT"; fi

            RAW_INF_OUTPUT=$(python3 calculate_inf.py "$native_exp_pdb_path" "$aa_out_path" 2>&1)
            INF_VAL=$(echo "$RAW_INF_OUTPUT" | grep "INF:" | awk '{print $2}')
            if [ -z "$INF_VAL" ]; then INF_VAL="Błąd_INF"; fi
        else
            RMSD_VAL="Brak_Ref"
            LDDT_VAL="Brak_Ref"
            INF_VAL="Brak_Ref"
        fi
    else
        RMSD_VAL="Błąd_Gen"
        LDDT_VAL="Błąd_Gen"
        INF_VAL="Błąd_Gen"
    fi
    
    echo " [WYNIK] $pdb_id | Seg: $segments | RMSD: $RMSD_VAL | lDDT: $LDDT_VAL | INF: $INF_VAL"
    
    printf "%-25s | %-3s | %-8s | %-6s | %-10s | %-15s | %-15s | %-10s\n" "$pdb_id" "$segments" "$sampler" "$steps" "${EXEC_TIME}" "$RMSD_VAL" "$LDDT_VAL" "$INF_VAL" >> "$METADATA"
}

STEPS_ARR=(500 750 1000)

# --- GŁÓWNA PĘTLA WYKONAWCZA ---
for ITEM in "${SELECTED_ITEMS[@]}"; do
    PDB_ID="${ITEM%:*}"
    SEGMENTS="${ITEM#*:}"

    REF_EXP_PDB="${REF_EXP_DIR}/${PDB_ID}.pdb" 
    DOTSEQ_FILE="${RESULTS_DIR}/${PDB_ID}.dotseq"

    echo "[*] Przetwarzanie: $PDB_ID (Liczba segmentów: $SEGMENTS)"

    echo "    -> Uruchamianie walidatora struktury dot-bracket..."
    VALIDATION_ERR=$(python3 rna_validator_standalone.py "$PDB_ID" "$REF_EXP_PDB" "$DOTSEQ_FILE" 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "    [!] UWAGA: Odrzucono strukturę."
        echo "        >> POWÓD: $VALIDATION_ERR"
        printf "%-25s | %-3s | %-8s | %-6s | %-10s | %-15s | %-15s | %-10s\n" "$PDB_ID" "$SEGMENTS" "odrzucono" "walidacja" "Błąd" "Błąd" "Błąd" "Błąd" >> "$METADATA"
        continue
    fi

    # ==========================================================
    # KROK 1: BASELINE (Domyślny solver 5000) vs NATYWNY
    # ==========================================================
    DDPM_CG_NAME="${PDB_ID}_default_5000_baseline_CG.pdb"
    DDPM_CG_PATH="${RESULTS_DIR}/${DDPM_CG_NAME}"
    DDPM_AA_PATH="${RESULTS_DIR}/${PDB_ID}_ddpm_5000_baseline_AA.pdb"

    if [ ! -f "$DDPM_AA_PATH" ]; then
        START_TIME_DDPM=$(date +%s)
        if [ ! -f "$DDPM_CG_PATH" ]; then
            grapharna --input="$DOTSEQ_FILE" --output-folder="$RESULTS_DIR" --output-name="$DDPM_CG_NAME" \
                      --seed=0 --timesteps=5000 --sampler="ddpm" --steps=5000
        fi
        if [ -f "$DDPM_CG_PATH" ]; then
            ./Arena/Arena "$DDPM_CG_PATH" "$DDPM_AA_PATH" 5
        fi
        END_TIME_DDPM=$(date +%s)
        EXEC_TIME_DDPM="$((END_TIME_DDPM - START_TIME_DDPM)) s"
    else
        EXEC_TIME_DDPM="Cache"
    fi

    if [ -f "$DDPM_AA_PATH" ]; then
        RAW_BASE_RMSD=$(python3 calculate_rmsd.py "$REF_EXP_PDB" "$DDPM_AA_PATH" 2>&1)
        BASE_RMSD=$(echo "$RAW_BASE_RMSD" | tail -n 1 | awk '{print $NF}')
        if ! [[ "$BASE_RMSD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then BASE_RMSD="Błąd"; fi
        
        RAW_BASE_LDDT=$(python3 compare.py --prediction "$DDPM_AA_PATH" --reference "$REF_EXP_PDB" 2>&1)
        BASE_LDDT=$(echo "$RAW_BASE_LDDT" | grep "Global Score:" | awk '{print $3}')
        if [ -z "$BASE_LDDT" ]; then BASE_LDDT="Błąd"; fi

        RAW_BASE_INF=$(python3 calculate_inf.py "$REF_EXP_PDB" "$DDPM_AA_PATH" 2>&1)
        BASE_INF=$(echo "$RAW_BASE_INF" | grep "INF:" | awk '{print $2}')
        if [ -z "$BASE_INF" ]; then BASE_INF="Błąd"; fi
        
        printf "%-25s | %-3s | %-8s | %-6s | %-10s | %-15s | %-15s | %-10s\n" "$PDB_ID" "$SEGMENTS" "default" "5000" "$EXEC_TIME_DDPM" "$BASE_RMSD" "$BASE_LDDT" "$BASE_INF" >> "$METADATA"
    fi

    # ==========================================================
    # KROK 2: KANAPKA vs NATYWNY
    # ==========================================================
    for steps in "${STEPS_ARR[@]}"; do
        run_test "$PDB_ID" "$SEGMENTS" "custom" "$steps" "quadratic" "$REF_EXP_PDB"
    done
    echo ""
done

# Czyszczenie śladów po Pythonie po zakończeniu eksperymentu
rm -f rna_validator_standalone.py

echo "=== Benchmark RMSD/lDDT/INF zakończony sukcesem! ==="
cat "$METADATA"