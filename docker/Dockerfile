ARG BASE_IMAGE=robotwin:cuda12.1
FROM ${BASE_IMAGE}

# Reuse the proven RoboTwin simulation dependency stack, but expose it as an
# independent TableSim image with its own user, home, cache, and workspace.
USER root

RUN groupmod --new-name tablesim robotwin \
    && usermod \
        --login tablesim \
        --home /home/tablesim \
        --move-home \
        --gid tablesim \
        robotwin \
    && mkdir -p /workspace/TableSim /home/tablesim/.cache \
    && chown -R tablesim:tablesim /workspace/TableSim /home/tablesim

USER tablesim
WORKDIR /workspace/TableSim

CMD ["bash"]
