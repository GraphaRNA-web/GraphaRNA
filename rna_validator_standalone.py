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
