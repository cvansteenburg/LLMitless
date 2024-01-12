# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit

# TODO:
# Run as root in dev to make sure it works
# Run with a user in dev to make sure it works
# Do I need to define poetry cache_dir if I just log in as user? Poetry will create in current user's home directory
# Copy that whole env dir into prod and it doesn't matter what poetry chose to name the venv
# In prod builder, log in as user and let poetry install create .cache

################################
# POETRY-ENV-BASE
# Provides common ENV variables and build args
################################

# Set base python image - use --build-arg at invocation to change
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} as poetry-env-base

ARG POETRY_VERSION=1.6.1

ENV POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_OPTIONS_ALWAYS_COPY=true

ENV PATH="$POETRY_HOME/bin:$PATH"

# print the path for debug
RUN echo $PATH

# ALSO UNCLEAR IF WE NEED PYTHONPATH


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

RUN echo $PATH


################################
# DEVELOPMENT
# Build a development container
################################

FROM poetry-base as development

ARG PROJECT_ROOT
ARG APPLICATION_SERVER_PORT
# A list of whatever tools you'll use in development. You may pass these from docker-compose.
# Pass these in with format --build-arg packages_to_install="package1 package2 package3"
ARG ADD_SYSTEM_LEVEL_PACKAGES

# ENV PYTHONPATH=/${PROJECT_ROOT} \ # TODO: Unclear if we need this, given venv isn't here
ENV APPLICATION_SERVER_PORT=$APPLICATION_SERVER_PORT

# Install dev dependencies
RUN apt-get install -y \
    git \
    $ADD_SYSTEM_LEVEL_PACKAGES \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo $PATH

# Note: manually specifying the path to the venv and adding it to the PATH eliminates
# the need to use poetry run before running a command in the dev container.
# I didn't do that because I want to stick with the default poetry behavior.

WORKDIR ${PROJECT_ROOT}

# ################################
# # PRODUCTION-DEP-BUILDER
# # Install deps and venv for production or CI image
# ################################

# FROM poetry-env-base as prod-dep-builder



# # Install production dependencies
# COPY ./pyproject.toml ./poetry.lock ./
# RUN --mount=type=cache,target=/root/.cache \
#     poetry install --only main --no-root

# ################################
# # PRODUCTION-BASE
# # Sets up the production base image
# ################################

# FROM ${PYTHON_IMAGE} as production-image

# ENV FASTAPI_ENV=production
# ENV PATH="/opt/poetry/bin:$PATH"

# COPY --from=poetry-base /opt/poetry /opt/poetry/
# COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
# COPY ./src /LLMitless/src
# WORKDIR /LLMitless/src

# # Principle of least privilege: create a new user for running the application
# RUN groupadd -g 1001 python_application && \
#     useradd -r -u 1001 -g python_application python_application

# # Set the WORKDIR to the application root.
# # https://www.uvicorn.org/settings/#development
# # https://docs.docker.com/engine/reference/builder/#workdir
# WORKDIR ${PYTHONPATH}
# RUN chown python_application:python_application ${PYTHONPATH}

# # Create cache directory and set permissions because user 1001 has no home
# # and poetry cache directory.
# # https://python-poetry.org/docs/configuration/#cache-directory
# RUN mkdir ${POETRY_CACHE_DIR} && chown python_application:python_application ${POETRY_CACHE_DIR}

# # Document the exposed port
# # https://docs.docker.com/engine/reference/builder/#expose
# EXPOSE ${APPLICATION_SERVER_PORT}

# # Use the unpriveledged user to run the application
# USER 1001

# # Run the uvicorn application server.
# CMD exec uvicorn --workers 1 --host 0.0.0.0 --port $APPLICATION_SERVER_PORT app.main:app

# # CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "80"]