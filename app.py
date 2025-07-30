from fastapi import FastAPI, Form, status
from fastapi.responses import PlainTextResponse, JSONResponse
import uuid
import os
import subprocess
from time import sleep

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OK"}

@app.post("/run")
async def run_grapharna(uuid: str = Form(...), seed: int = Form(42)):
    """
    This function is the main FastAPI endpoint. It takes 2 parameters: uuid and seed.
    First param is used to identify the job and locate necessery files in the shared volume.
    Second param is the seed engine parameter.

    Processing:
    A shared volume is used between backend and the engine, in order to minimise necessery http data transfers
    #1 Folder setup: we user 2 distinct folders: engine inputs, where backend creates the files and engine outputs
    #2 we use the subprocess run command in order to calculate the .pdb results
    #3 we check if the file was generated
    #4 we run the annotator subprocess in order to generate output in fasta format
    """

    input_path = f"/shared/samples/engine_inputs/{uuid}.dotseq"
    output_folder = f"/shared/samples/engine_outputs"
    output_name = uuid

    output_path_pdb = os.path.join(output_folder, output_name + ".pdb")
    output_path_json = os.path.join(output_folder, output_name + ".json")
    output_path_dot = os.path.join(output_folder, output_name + ".dot")


    try:
        subprocess.run([
            "grapharna",
            f"--input={input_path}",
            f"--seed={seed}",
            f"--output-folder={output_folder}",
            f"--output-name={output_name}"
        ], check=True)

        for _ in range(20):
            if os.path.exists(output_path_pdb):
                break
            sleep(0.5)

        if not os.path.exists(output_path_pdb):
            print(f"Output file {output_path_pdb} can't be found or wasn't generated.")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": f"Output file {output_path_pdb} can't be found or wasn't generated."}
            )
        
        try:
            subprocess.run([
                "annotator",
                "--json", str(output_path_json),
                f"--dot",  str(output_path_dot),
                f"--extended", str(output_path_pdb)
            ], check=True)
        
        except subprocess.CalledProcessError as e:
            print(f"Annotator has failed")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"ERROR": f"Annotator has failed"}
            )
        
            
        return_content = {"message": "OK", "pdbFilePath": output_path_pdb, 
                          "jsonFilePath": output_path_json, "dotFilePath": output_path_dot}
        
        return JSONResponse(content=return_content, status_code=status.HTTP_200_OK)

    except subprocess.CalledProcessError as e:
        print(f"GraphaRNA engine failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ERROR": f"GraphaRNA engine has failed"}
        )
    

@app.post("/test")
async def test_stub():
    seed = 42
    input_path = "/shared/samples/engine_inputs/test.dotseq"
    with open(input_path, "r") as f:
        tekst = f.read()

    output_dir = f"/shared/samples/engine_outputs"
    output_path = f"/shared/samples/engine_outputs/test.pdb"

    os.makedirs(output_dir, exist_ok=True)

    # Symulacja obliczeń
    sleep(5)

    with open(output_path, "w") as f:
        f.write(tekst)

    return PlainTextResponse(output_path)