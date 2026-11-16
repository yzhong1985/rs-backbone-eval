# syntax=docker/dockerfile:1.7
#
# One image with every worker environment. torch is the bulk of it and the
# workers pin four different builds (clay 2.4.0+cu121, lim/olmoearth
# 2.9.1+cu128, anysat/prithvi 2.14.1+cu130, aef 2.15.0+cpu), so only the pairs
# that share a wheel can share bytes. UV_LINK_MODE=hardlink plus a single RUN
# keeps one copy of each file inside the layer: lim and olmoearth share their
# torch, anysat and prithvi share theirs, and the image lands around 20 GB
# instead of ~30 GB. Never split this RUN or COPY a venv in: both copy data.
#
# Data, weights, caches and the run ledger are NOT baked in; mount them (see
# compose.yaml). The image only carries code, environments and tooling.

ARG UV_VERSION=0.13.0
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ubuntu:24.04

ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=hardlink \
    UV_CACHE_DIR=/home/rs/.cache/uv \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH=/home/rs/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git libglib2.0-0 libgl1 libgomp1 \
      passwd python3.12 python3.12-venv tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# ubuntu:24.04 already ships uid/gid 1000 as the `ubuntu` user, so the build
# must reuse it instead of failing on a duplicate id, while still honouring a
# host that uses another uid.
RUN if getent group "${GID}" > /dev/null; then \
        groupmod -n rs "$(getent group "${GID}" | cut -d: -f1)"; \
    else groupadd -g "${GID}" rs; fi; \
    if getent passwd "${UID}" > /dev/null; then \
        usermod -l rs -d /home/rs -m "$(getent passwd "${UID}" | cut -d: -f1)"; \
        usermod -g "${GID}" rs; \
    else useradd -m -u "${UID}" -g "${GID}" -s /bin/bash rs; fi; \
    mkdir -p /home/rs \
 && chown -R "${UID}:${GID}" /home/rs

USER rs
WORKDIR /app

# The cache lives under /home/rs (not /opt): removing a directory needs write
# permission on its parent, and /opt is root-owned. Order matters: the first
# uv sync populates the cache, later ones
# hardlink out of it, so everything must happen before the cache is deleted.
RUN uv python install 3.12
COPY --chown=rs:rs pyproject.toml uv.lock README.md ./
COPY --chown=rs:rs src ./src
COPY --chown=rs:rs configs ./configs
COPY --chown=rs:rs scripts ./scripts
COPY --chown=rs:rs tests ./tests
COPY --chown=rs:rs workers ./workers
COPY --chown=rs:rs web ./web
# Hosts in mainland China reach PyPI at ~15 kB/s and cannot reach
# download.pytorch.org at all, so the build rewrites the lock URLs to mirrors
# that serve byte-identical files (the lock hashes still verify). Pass empty
# build args to build against the upstream hosts.
ARG PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/
ARG TORCH_MIRROR=https://mirrors.aliyun.com/pytorch-wheels/
RUN if [ -n "${PYPI_MIRROR}" ]; then \
        sed -i "s#https://files.pythonhosted.org/#${PYPI_MIRROR}#g" uv.lock workers/*/uv.lock; \
    fi; \
    if [ -n "${TORCH_MIRROR}" ]; then \
        sed -i "s#https://download.pytorch.org/whl/#${TORCH_MIRROR}#g" workers/*/uv.lock; \
    fi
RUN uv sync --frozen --extra portal --extra downloads --extra dev \
 && for worker in aef anysat clay lim olmoearth prithvi; do \
        uv sync --frozen --project "workers/${worker}"; \
    done \
 && rm -rf "$UV_CACHE_DIR"

# Worker source trees (CLAY_SRC / ANYSAT_SRC) are pinned upstream checkouts and
# are mounted, not built in; the registries read them from the environment.
ENV RS_EVAL_CONFIG_ROOT=/app/configs \
    RS_EVAL_DATA_ROOT=/mnt/data \
    RS_EVAL_WEIGHTS_ROOT=/mnt/weights \
    RS_EVAL_RUNS_ROOT=/mnt/runs \
    RS_EVAL_CACHE_ROOT=/mnt/cache \
    RS_EVAL_QUEUE_ROOT=/mnt/runs/../queue \
    PYTHONPATH=/app/src

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uv", "run", "rs-eval", "doctor"]
