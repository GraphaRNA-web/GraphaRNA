FROM python:3.11-slim

WORKDIR /app

# Install sys requirements
RUN apt-get update && apt-get install -y wget graphviz build-essential clang && rm -rf /var/lib/apt/lists/*

# Copy rarely changing python dependencies
COPY pyproject.toml docker_requirements.txt ./

# Python tools&requirements installation
# Source: https://docs.docker.com/build/cache/optimize/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel build \
    && pip install -r docker_requirements.txt

# Copy only the dependency library
COPY RiNALMo /app/RiNALMo
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ./RiNALMo


# Install the Arena tool
COPY Arena /app/Arena
RUN cd /app/Arena && make Arena
ENV PATH="/app/Arena:${PATH}"

# Finally copy the source code 
COPY . .
RUN pip install . --no-deps


# Otwórz port FastAPI
EXPOSE 8080

# Domyślnie uruchamiaj serwer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
