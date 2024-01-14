# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit

################################
# POETRY-ENV-BASE
# Set common ENV variables args, and build virtual env
################################

# Set base python image - use --build-arg at invocation to change
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} as poetry-env-base

ARG POETRY_VERSION=1.6.1
ARG PROJECT_ROOT

ENV POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME="/opt/poetry" \
    # Define poetry cache location, so we can attach Docker cache to it later for fast installs
    POETRY_CACHE_DIR="/opt/poetry_cache/.cache/pypoetry/" \
    # Where our project's virtual environment will live
    VIRTUAL_ENV="/venv"
# Prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
# Prepend project root to the path that Python uses to locate packages
ENV PYTHONPATH="/${PROJECT_ROOT}:${PYTHONPATH}"
# Prepare project's virtual env
RUN python -m venv $VIRTUAL_ENV

################################
# POETRY-BASE
# Install Poetry for use in our virtual environment
################################

FROM poetry-env-base as poetry-base

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Update pip and install tools required to install Poetry
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential

# https://python-poetry.org/docs/#installing-manually
RUN python -m venv ${POETRY_HOME}
RUN ${POETRY_HOME}/bin/pip install -U pip setuptools
RUN ${POETRY_HOME}/bin/pip install "poetry==${POETRY_VERSION}"

# Install production dependencies
WORKDIR ${PROJECT_ROOT}
COPY pyproject.toml poetry.lock ./
# Deps are installed in {POETRY_CACHE_DIR}, so we'll mount a Docker cache there
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --only main --no-root


################################
# DEVELOPMENT
# Build a development container
################################

FROM poetry-base as development

ARG APPLICATION_DEV_SERVER_PORT
# A list of whatever tools you'll use in development. You may pass these from docker-compose.
# Pass these in with format --build-arg packages_to_install="package1 package2 package3"
ARG ADD_SYSTEM_LEVEL_PACKAGES

ENV DEVELOPMENT_STAGE=DEVELOPMENT
ENV APPLICATION_DEV_SERVER_PORT=$APPLICATION_DEV_SERVER_PORT

# Install dev dependencies
RUN apt-get install -y \
    git \
    $ADD_SYSTEM_LEVEL_PACKAGES \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${PROJECT_ROOT}

# Install dev dependencies. 
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --without main --no-root

EXPOSE ${APPLICATION_DEV_SERVER_PORT}

CMD ["bash"]

# Optional: start uvicorn. Alternatively, start it from Compose, devcontainer.json etc
# CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "80"]


################################
# PRODUCTION
# Sets up the production image
################################

FROM poetry-env-base as production

ARG APPLICATION_PROD_SERVER_PORT

ENV DEVELOPMENT_STAGE=PRODUCTION

# Copy in our built venv which has our deps
COPY --from=poetry-base-base $VIRTUAL_ENV $VIRTUAL_ENV

WORKDIR ${PROJECT_ROOT}
# Copy in the core application code
COPY src .

# Principle of least privilege: create a new user for running the application
RUN groupadd -g 1001 python_application && \
    useradd -r -u 1001 -g python_application python_application

# Grant access to the project root
RUN chown python_application:python_application ${PROJECT_ROOT}

EXPOSE ${APPLICATION_PROD_SERVER_PORT}

# Use the unprivileged user to run the application
USER 1001

################################
# TEST-IMAGE
# Sets up the test image
################################

FROM poetry-env-base as test-image

ARG APPLICATION_TEST_SERVER_PORT

ENV DEVELOPMENT_STAGE=TESTING

WORKDIR ${PROJECT_ROOT}
COPY tests src ./

# Principle of least privilege: create a new user for running the application
RUN groupadd -g 1001 python_application && \
    useradd -r -u 1001 -g python_application python_application

# Grant access to the project root
RUN chown python_application:python_application ${PROJECT_ROOT}

EXPOSE ${APPLICATION_TEST_SERVER_PORT}

USER 1001