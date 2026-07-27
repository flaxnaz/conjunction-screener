"""Space-Track authentication.

Wraps credential loading (from environment variables / a local .env file)
and constructs an authenticated `spacetrack.SpaceTrackClient`.

Credentials are never hard-coded and never logged. Locally they come from
a `.env` file (see `.env.example`); in CI they come from GitHub secrets
injected as environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from spacetrack import SpaceTrackClient


class MissingCredentialsError(RuntimeError):
    """Raised when Space-Track credentials cannot be found."""


@dataclass(frozen=True)
class SpaceTrackCredentials:
    identity: str
    password: str


def load_credentials(dotenv_path: str | None = None) -> SpaceTrackCredentials:
    """Load Space-Track credentials from the environment.

    Looks for SPACETRACK_IDENTITY and SPACETRACK_PASSWORD. Calls
    `load_dotenv` first so a local `.env` file (git-ignored) is picked up
    automatically; in CI the variables are already present via secrets,
    so the missing `.env` file is a harmless no-op.
    """
    load_dotenv(dotenv_path)

    identity = os.getenv("SPACETRACK_IDENTITY")
    password = os.getenv("SPACETRACK_PASSWORD")

    if not identity or not password:
        raise MissingCredentialsError(
            "SPACETRACK_IDENTITY and SPACETRACK_PASSWORD must be set "
            "(via a local .env file or environment variables)."
        )

    return SpaceTrackCredentials(identity=identity, password=password)


def get_client(dotenv_path: str | None = None) -> SpaceTrackClient:
    """Return an authenticated Space-Track client.

    The `spacetrack` library authenticates lazily on first request, so
    this does not make a network call by itself.
    """
    creds = load_credentials(dotenv_path)
    return SpaceTrackClient(identity=creds.identity, password=creds.password)
