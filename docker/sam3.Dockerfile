ARG BASE_IMAGE=tablesim:cuda12.1
FROM ${BASE_IMAGE}

ARG SAM3_COMMIT=46957e47805eaa273f4aa7bbbd25a88bca9108ce

LABEL org.opencontainers.image.title="TableSim SAM3 perception sidecar" \
      org.opencontainers.image.source="https://github.com/facebookresearch/sam3" \
      org.opencontainers.image.revision="${SAM3_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/sam3-venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/tablesim/.cache/huggingface \
    PYTHONPATH=/workspace/TableSim:/opt/TableSim

USER root

# Reuse the already validated TableSim CUDA/PyTorch runtime, but install SAM3
# into a separate venv. The original tablesim:cuda12.1 image remains unchanged.
RUN python -m venv --system-site-packages /opt/sam3-venv \
    && /opt/sam3-venv/bin/python -m pip install --upgrade pip "setuptools>=70,<81" wheel \
    && /opt/sam3-venv/bin/python -m pip install \
        "einops>=0.8,<1" \
        "ftfy==6.1.1" \
        "regex" \
        "timm>=1.0.17,<2"

# Upstream imports its COCO loader from the image-model import chain even when
# training is disabled, so this small runtime wheel is required for inference.
RUN python -m pip install "pycocotools>=2.0.7,<3"
RUN python -m pip install "psutil>=5.9,<8"

COPY high_level_planner/third_party/sam3/pyproject.toml \
     high_level_planner/third_party/sam3/README.md \
     high_level_planner/third_party/sam3/LICENSE \
     high_level_planner/third_party/sam3/MANIFEST.in \
     /tmp/sam3-src/
COPY high_level_planner/third_party/sam3/sam3 /tmp/sam3-src/sam3

RUN python -m pip install --no-build-isolation --no-deps /tmp/sam3-src \
    && rm -rf /tmp/sam3-src

RUN mkdir -p /home/tablesim/.cache/huggingface /opt/TableSim \
    && chown -R tablesim:tablesim /home/tablesim /opt/TableSim

COPY --chown=tablesim:tablesim high_level_planner /opt/TableSim/high_level_planner

USER tablesim
WORKDIR /workspace/TableSim

EXPOSE 8003

CMD ["python", "-m", "high_level_planner.perception.sam3_server", "--host", "0.0.0.0", "--port", "8003"]
