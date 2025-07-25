from fastapi import FastAPI, UploadFile, Form, status
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
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

    EPOCHS = 800

    input_path = f"/shared/samples/grapharna-seed={seed}/{EPOCHS}/{uuid}.dotseq"
    output_folder = f"/shared/samples/grapharna-seed={seed}/{EPOCHS}"
    output_name = uuid

    output_path = os.path.join(output_folder, output_name + ".pdb")

    try:
        subprocess.run([
            "grapharna",
            f"--input={input_path}",
            f"--seed={seed}",
            f"--output-folder={output_folder}",
            f"--output-name={output_name}"
        ], check=True)

        for _ in range(20):
            if os.path.exists(output_path):
                break
            sleep(0.5)

        if not os.path.exists(output_path):
            print(f"Output file {output_path} can't be found or wasn't generated.")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": f"Output file {output_path} can't be found or wasn't generated."}
            )
            

        return FileResponse(output_path, media_type="text/plain", filename=f"{uuid}.pdb")

    except subprocess.CalledProcessError as e:
        return {"error": f"Grapharna failed: {e}"}
    

@app.post("/test")
async def test_stub():
    seed = 42
    input_path = "/shared/user_inputs/test.dotseq"
    with open(input_path, "r") as f:
        tekst = f.read()

    output_dir = f"/shared/samples/grapharna-seed={seed}/800/"
    output_path = f"/shared/samples/grapharna-seed={seed}/800/test.pdb"

    os.makedirs(output_dir, exist_ok=True)

    # Symulacja obliczeń
    sleep(5)

    with open(output_path, "w") as f:
        f.write(tekst)

    return PlainTextResponse(output_path)