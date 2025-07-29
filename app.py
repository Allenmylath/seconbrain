#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Real Estate Bot Modal Example - OpenAI + Cartesia.

This module shows how to deploy a real estate search bot using Modal and FastAPI.
Uses OpenAI LLM + Cartesia TTS with property search capabilities.

It includes:
- FastAPI endpoints for starting agents and checking bot statuses.
- OpenAI LLM with Cartesia TTS integration.
- Real estate property search functionality.
- Use of a Daily transport for bot communication.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Container specifications for the FastAPI web server
web_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi", "uvicorn[standard]", "pydantic", "aiohttp")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily]")
    .add_local_dir("src", remote_path="/root/src")
)

# Container specifications for the Pipecat pipeline
bot_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily,openai,silero,cartesia,deepgram]")
    .pip_install("pymongo", "python-dotenv", "loguru")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("secondbrain_realestate", secrets=[modal.Secret.from_dotenv()])

# Global variables for bot management
bot_jobs = {}
daily_helpers = {}


# Define models at module level
class ConnectData(BaseModel):
    """Data provided by client to specify the services.

    This matches the format expected by the React frontend.
    """

    services: Dict[str, str] = {"llm": "openai", "tts": "cartesia"}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for function_call in bot_jobs.values():
        # function_call is already a FunctionCall object, just cancel it directly
        if function_call:
            try:
                function_call.cancel()
            except Exception as e:
                print(f"Error cancelling function: {e}")


async def create_room_and_token() -> tuple[str, str]:
    """Create a Daily room and generate an authentication token.

    This function checks for existing room URL and token in the environment variables.
    If not found, it creates a new room using the Daily API and generates a token for it.

    Returns:
        tuple[str, str]: A tuple containing the room URL and the authentication token.

    Raises:
        HTTPException: If room creation or token generation fails.
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRoomParams

    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    if not room_url:
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")
        room_url = room.url

        token = await daily_helpers["rest"].get_token(room_url)
        if not token:
            raise HTTPException(
                status_code=500, detail=f"Failed to get token for room: {room_url}"
            )

    return room_url, token


async def start():
    """Internal method to start the OpenAI + Cartesia real estate bot agent.

    Returns:
        tuple[str, str]: A tuple containing the room URL and token.
    """
    room_url, token = await create_room_and_token()
    launch_bot_func = modal.Function.from_name("secondbrain_realestate", "bot_runner")
    function_call = launch_bot_func.spawn(room_url, token)

    # Store the FunctionCall object directly, with room_url as key for identification
    bot_jobs[room_url] = function_call

    return room_url, token


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


@app.function(
    image=bot_image,
    keep_warm=1,
    allow_concurrent_inputs=1,
    timeout=600,  # 5 minutes timeout
)
async def bot_runner(room_url: str, token: str):
    """Launch the OpenAI + Cartesia real estate bot process.

    Args:
        room_url (str): The URL of the Daily room where the bot and client will communicate.
        token (str): The authentication token for the room.

    Raises:
        HTTPException: If the bot pipeline fails to start.
    """
    try:
        # Import the bot runner
        from src.bot import run_bot

        print(f"Starting OpenAI + Cartesia real estate bot: -u {room_url} -t {token}")
        await run_bot(room_url, token)
    except Exception as e:
        print(f"Bot pipeline error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start bot pipeline: {e}"
        )


@app.function(
    image=web_image,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """
    # Initialize FastAPI app
    web_app = FastAPI(
        title="Pipecat Real Estate Bot",
        description="OpenAI + Cartesia Real Estate Search Bot API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define routes directly in the FastAPI app
    '''
    @web_app.get("/")
    async def start_agent():
        """A user endpoint for launching a bot agent and redirecting to the created room URL.

        This function starts the OpenAI + Cartesia real estate bot agent and redirects the user
        to the room URL to interact with the bot through a Daily Prebuilt Interface.

        Returns:
            RedirectResponse: A response that redirects to the room URL.
        """
        print("Starting OpenAI + Cartesia real estate bot")
        room_url, token = await start()
        return RedirectResponse(room_url)
'''

    @web_app.post("/connect")
    async def rtvi_connect(data: ConnectData = None) -> Dict[Any, Any]:
        """A user endpoint for launching a bot agent and retrieving the room/token credentials.

        This function starts the OpenAI + Cartesia real estate bot agent and returns the room URL
        and token for the bot. This allows the client to then connect to the bot using
        their own RTVI interface.

        Args:
            data (ConnectData): Optional. The data containing the services to use.

        Returns:
            Dict[Any, Any]: A dictionary containing the room URL and token.
        """
        print("Starting OpenAI + Cartesia real estate bot from /connect endpoint")
        if data and data.services:
            print(f"Services requested: {data.services}")

            # Validate that the frontend is requesting the correct services
            if (
                data.services.get("llm") != "openai"
                or data.services.get("tts") != "cartesia"
            ):
                raise HTTPException(
                    status_code=400,
                    detail="This backend only supports OpenAI LLM and Cartesia TTS",
                )

        room_url, token = await start()
        return {"room_url": room_url, "token": token}

    @web_app.get("/status/{fid}")
    def get_status(fid: str):
        """Retrieve the status of a bot process by its function ID.

        Args:
            fid (str): The function ID of the bot process.

        Returns:
            JSONResponse: A JSON response containing the bot's status and result code.

        Raises:
            HTTPException: If the bot process with the given ID is not found.
        """
        func = modal.FunctionCall.from_id(fid)
        if not func:
            raise HTTPException(
                status_code=404, detail=f"Bot with process id: {fid} not found"
            )

        try:
            result = func.get(timeout=0)
            return JSONResponse({"bot_id": fid, "status": "finished", "code": result})
        except modal.exception.OutputExpiredError:
            return JSONResponse({"bot_id": fid, "status": "finished", "code": 404})
        except TimeoutError:
            return JSONResponse({"bot_id": fid, "status": "running", "code": 202})

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return web_app
