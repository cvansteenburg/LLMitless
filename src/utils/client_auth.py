import os
import secrets
from logging import getLogger
from typing import Annotated

import sentry_sdk
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = getLogger(__name__)

async def security():
    try:
        basic_security = HTTPBasic()
    except Exception as e:
        sentry_sdk.capture_exception(e)

    return basic_security


BasicCreds = Annotated[HTTPBasicCredentials, Depends(security)]


# Basic security until we implement more robust auth
async def check_basic_auth(credentials: BasicCreds) -> bool:
    BASIC_AUTH_USERNAME = os.getenv("BASIC_AUTH_USERNAME")
    BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")

    sentry_sdk.capture_message(f"BASIC_AUTH_USERNAME: {BASIC_AUTH_USERNAME}, BASIC_AUTH_PASSWORD: {BASIC_AUTH_PASSWORD}")

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
            
            sentry_sdk.capture_message(f"Incorrect username or password: {credentials.username} - {credentials.password}, CORRECT USERNAME: {correct_username}, CORRECT PASSWORD: {correct_password}")


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