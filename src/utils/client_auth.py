import os
import secrets
from logging import getLogger
from typing import Annotated

import sentry_sdk
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = getLogger(__name__)

basic_security = HTTPBasic()

BasicCreds = Annotated[HTTPBasicCredentials, Depends(basic_security)]


# Basic security until we implement more robust auth
async def check_basic_auth(credentials: BasicCreds) -> bool:
    BASIC_AUTH_USERNAME = os.getenv("BASIC_AUTH_USERNAME")
    BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")

    # Set Sentry context to capture the actual values from the environment
    sentry_sdk.set_context("environment_variables", {
        "BASIC_AUTH_USERNAME": BASIC_AUTH_USERNAME,
        "BASIC_AUTH_PASSWORD": BASIC_AUTH_PASSWORD,
    })

    UNAUTH_EXC = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )

    try:
        if BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD:
            correct_username = secrets.compare_digest(
                credentials.username.encode("utf8"), BASIC_AUTH_USERNAME.encode("utf8")
            )
            correct_password = secrets.compare_digest(
                credentials.password.encode("utf8"), BASIC_AUTH_PASSWORD.encode("utf8")
            )
            if correct_username and correct_password:
                return True

        raise UNAUTH_EXC

    except HTTPException:
        raise

    except Exception as e:
        exception_type = type(e).__name__
        logger.error(f"Exception in simple auth: {exception_type}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error",
        )