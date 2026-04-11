FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    clang \
    && rm -rf /var/lib/apt/lists/*


RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml docker_requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel build \
    && pip install -r docker_requirements.txt

COPY RiNALMo ./RiNALMo
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ./RiNALMo

COPY Arena ./Arena
RUN cd ./Arena && make Arena

COPY src ./src
RUN pip install . --no-deps

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /build/Arena /app/Arena
ENV PATH="/app/Arena:${PATH}"

COPY . .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]