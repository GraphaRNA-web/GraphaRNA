import json
from fastapi import FastAPI, Form, status, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
import uuid
import os
import subprocess
from time import sleep

def run_engine_background(uuid, seed, input_path, output_folder, output_name, output_path_pdb, output_path_json, error_path):
    
    try:
        subprocess.run([
            "grapharna",
            f"--input={input_path}",
            f"--seed={seed}",
            f"--output-folder={output_folder}",
            f"--output-name={output_name}"
        ], check=True)

        if not os.path.exists(output_path_pdb):
            raise Exception("GraphaRNA finished but output PDB is missing")

        subprocess.run([
            "Arena",
            output_path_pdb,
            output_path_pdb,
            "5"
        ], check=True, capture_output=True)

        subprocess.run([
            "annotator",
            "--json", str(output_path_json),
            "--extended", str(output_path_pdb)
        ], check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        error_data = {"error": "Process failed", "cmd": e.cmd, "stderr": e.stderr.decode() if e.stderr else ""}
        with open(error_path, "w") as f:
            json.dump(error_data, f)
        print(f"Background task failed: {e}")

    except Exception as e:
        error_data = {"error": str(e)}
        with open(error_path, "w") as f:
            json.dump(error_data, f)
        print(f"Background task failed: {e}")

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OK"}

    
@app.post("/run")
async def run_grapharna(
    background_tasks: BackgroundTasks,
    uuid: str = Form(...), 
    seed: int = Form(42)
):
    print(f"Incoming request with uuid: {uuid} and seed: {seed}")
    
    input_path = f"/shared/samples/engine_inputs/{uuid}.dotseq"
    output_folder = "/shared/samples/engine_outputs"
    output_name = f"{uuid}_{seed}"
    
    output_path_pdb = os.path.join(output_folder, output_name + ".pdb")
    output_path_json = os.path.join(output_folder, output_name + ".json")
    error_path = os.path.join(output_folder, output_name + ".err")

    if not os.path.exists(input_path):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Input file {input_path} does not exist."}
        )

    if os.path.exists(error_path): os.remove(error_path)
    if os.path.exists(output_path_json): os.remove(output_path_json)

    background_tasks.add_task(
        run_engine_background,
        uuid, seed, input_path, output_folder, output_name, 
        output_path_pdb, output_path_json, error_path
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": "Job accepted", 
            "status_endpoint": f"/status/{uuid}"
        }
    )

@app.get("/status/{uuid}")
async def check_status(uuid: str, seed: int):
    output_folder = "/shared/samples/engine_outputs"
    output_name = f"{uuid}_{seed}"
    
    output_path_pdb = os.path.join(output_folder, output_name + ".pdb")
    output_path_json = os.path.join(output_folder, output_name + ".json")
    error_path = os.path.join(output_folder, output_name + ".err")
    print(f"output_path_pdb: {output_path_pdb}, output_path_json: {output_path_json}, error_path: {error_path}")
    if os.path.exists(error_path):
        with open(error_path, "r") as f:
            err_content = json.load(f)
        try:
            os.remove(error_path)
        except OSError as e:
            print(f"Error removing file {error_path}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err_content
        )

    if os.path.exists(output_path_json) and os.path.exists(output_path_pdb):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "COMPLETED",
                "pdbFilePath": output_path_pdb,
                "jsonFilePath": output_path_json
            }
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"status": "PROCESSING"}
    )


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