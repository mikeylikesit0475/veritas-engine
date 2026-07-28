FROM python:3.12-slim

# A minimal Python image for the veritas sandbox (ARCH §5).
# The ContainerCodeExecutor runs `python3 -c <code>` inside this image with:
#   - network disabled
#   - all Linux capabilities dropped
#   - no-new-privileges
# No pip packages are installed — the quant agent writes pure-Python computations
# (arithmetic, statistics). If a claim needs a library, the agent should reason
# about it in prose, not import it.

# Keep the image small and fast to start.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for defence in depth (capabilities are already dropped, but
# this is belt-and-braces).
RUN useradd --create-home --shell /bin/false sandbox
USER sandbox
WORKDIR /home/sandbox