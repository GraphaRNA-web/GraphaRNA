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
    print(f"Incomming request with uuid: {uuid} and seed: {seed}")
    input_path = f"/shared/samples/engine_inputs/{uuid}.dotseq"
    output_folder = f"/shared/samples/engine_outputs"
    output_name = f"{uuid}_{seed}"

    output_path_pdb = os.path.join(output_folder, output_name + ".pdb")
    output_path_json = os.path.join(output_folder, output_name + ".json")

    try:
        subprocess.run([
            "grapharna",
            f"--input={input_path}",
            f"--seed={seed}",
            f"--output-folder={output_folder}",
            f"--output-name={output_name}"
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"GraphaRNA engine failed. Stderr: {e.stderr}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ERROR": "GraphaRNA engine has failed", "details": e.stderr}
        )
    
    for _ in range(20):
            if os.path.exists(output_path_pdb):
                break
            sleep(0.5)

    if not os.path.exists(output_path_pdb):
        print(f"GraphaRNA did not generate the output file: {output_path_pdb}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "GraphaRNA finished without error, but the output file is missing."}
        )
        
    try:
        subprocess.run([
            "Arena",
            output_path_pdb,
            output_path_pdb,
            "5"
        ], check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"Arena conversion failed. Stderr: {e.stderr}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ERROR": "Arena conversion has failed", "details": e.stderr}
        )
        
    try:
        subprocess.run([
            "annotator",
            "--json", str(output_path_json),
            "--extended", str(output_path_pdb)
        ], check=True, capture_output=True, text=True)
    
    except subprocess.CalledProcessError as e:
        print(f"Annotator has failed. Stderr: {e.stderr}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ERROR": "Annotator has failed", "details": e.stderr}
        )
        
    return_content = {
        "message": "OK", 
        "pdbFilePath": output_path_pdb, 
        "jsonFilePath": output_path_json
    }
    
    return JSONResponse(content=return_content, status_code=status.HTTP_200_OK)

@app.post("/test")
async def test_run(uuid: str = Form(...), seed: int = Form(42)):
    """
    This function is the testing function that the backend can use. Takes the same params as the run function,
    so it can be used by changing the /run to /test in tasks.py. It behaves exactly the same as the /run endpoint, but
    it does not turn the subprocess on, saving about 30-40min per test. The output is a random .pdb file that has to be
    placed in the main engine folder under the name "test_res.pdb" BEFORE the image is built
    """
    print(f"Incomming request with uuid: {uuid} and seed: {seed}")
    output_folder = f"/shared/samples/engine_outputs"
    output_name = f"{uuid}_{seed}"

    output_path_pdb = os.path.join(output_folder, output_name + ".pdb")
    output_path_json = os.path.join(output_folder, output_name + ".json")

    test_path = "test_res.pdb"

    try:
        with open(test_path, "r") as f:
            tekst = f.readlines()
        with open(output_path_pdb, "w") as f:
            f.writelines(tekst)
        sleep(1)

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
                "Arena",
                output_path_pdb,
                output_path_pdb,
                "5"
            ], check=True, capture_output=True, text=True)

        except subprocess.CalledProcessError as e:
            print(f"Arena conversion failed. Stderr: {e.stderr}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"ERROR": "Arena conversion has failed", "details": e.stderr})
        
        try:
            result = subprocess.run([
                "annotator",
                "--json", str(output_path_json),
                "--extended", str(output_path_pdb)
            ], check=True, stderr=subprocess.PIPE)

        
        except subprocess.CalledProcessError as e:
            print(f"Annotator has failed, {e.stderr.decode()}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"ERROR": f"Annotator has failed"}
            )
        
            
        return_content = {"message": "OK", "pdbFilePath": output_path_pdb, 
                          "jsonFilePath": output_path_json}
        
        return JSONResponse(content=return_content, status_code=status.HTTP_200_OK)

    except subprocess.CalledProcessError as e:
        print(f"GraphaRNA engine failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ERROR": f"GraphaRNA engine has failed"}
        )