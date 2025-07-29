#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import uuid
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger
from strands import Agent, tool


from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
    RTVIObserverParams,
)
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

import openai
from pymongo import MongoClient
import re
import time
import traceback

load_dotenv(override=True)

# Set up OpenAI client for embeddings
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB connection string
MONGODB_CONNECTION_STRING = "mongodb+srv://allengeorgemylath:Mylath%4090@cluster0.jy2twdh.mongodb.net/real_estate"


async def send_rtvi_message(rtvi_instance, search_data: Dict[str, Any], text_query: str):
    """Send RTVI message - async helper function."""
    try:
        if not rtvi_instance:
            logger.warning("⚠️ RTVI processor not available")
            return

        if search_data.get("search_completed") and search_data.get("properties"):
            # Create structured message for successful search
            rtvi_message_data = {
                "type": "property_search_results",
                "timestamp": time.time(),
                "search_id": str(uuid.uuid4()),
                "query": text_query,
                "summary": {
                    "total_found": search_data.get("results_found", 0),
                    "showing": len(search_data["properties"]),
                    "execution_time": search_data.get("execution_time_seconds", 0),
                    "search_type": search_data.get("search_type", "hybrid"),
                },
                "filters_applied": search_data.get("filters_applied", {}),
                "properties": [
                    {
                        "id": prop["property_id"],
                        "url": prop["url"],
                        "images": {
                            "primary": prop["primary_image"],
                            "all": prop["image_urls"],
                        },
                        "details": {
                            "address": prop["address"],
                            "price": prop["price"],
                            "currency": prop["currency"],
                            "bedrooms": prop["bedrooms"],
                            "bathrooms": prop["bathrooms"],
                            "type": prop["property_type"],
                            "description": prop["description"],
                        },
                        "metadata": {
                            "search_score": prop["search_score"],
                            "mls_genuine": prop["mls_genuine"],
                            "status": prop["status"],
                        },
                    }
                    for prop in search_data["properties"]
                ],
            }

            # Send RTVI message with timeout to prevent blocking
            server_message_frame = RTVIServerMessageFrame(data=rtvi_message_data)

            try:
                await asyncio.wait_for(
                    rtvi_instance.push_frame(server_message_frame), timeout=5.0
                )
                logger.info(
                    f"✅ RTVI message sent from search - {len(search_data['properties'])} properties"
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ RTVI message send timed out")

        else:
            # Send error message
            rtvi_message_data = {
                "type": "property_search_error",
                "timestamp": time.time(),
                "search_id": str(uuid.uuid4()),
                "query": text_query,
                "error": search_data.get("error", "No properties found"),
            }

            server_message_frame = RTVIServerMessageFrame(data=rtvi_message_data)
            try:
                await asyncio.wait_for(
                    rtvi_instance.push_frame(server_message_frame), timeout=3.0
                )
                logger.info("✅ RTVI error message sent from search")
            except asyncio.TimeoutError:
                logger.warning("⚠️ RTVI error message send timed out")

    except Exception as e:
        logger.error(f"❌ Error sending RTVI message from search: {e}")


# Global variables for RTVI injection
global_rtvi_instance = None
global_event_loop = None


@tool
def execute_hybrid_search(
    text_query: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[str] = None,
    bathrooms: Optional[str] = None,
    property_type: Optional[str] = None,
    location_keywords: Optional[str] = None,
    mls_genuine: Optional[bool] = None,
    limit: int = 10,
    vector_search_index: str = "vector_index",
) -> Dict[str, Any]:
    """
    Execute hybrid search combining vector similarity with traditional filters.
    Updated to match MongoDB collection structure with nested property_details.
    Includes RTVI messaging via run_coroutine_threadsafe.
    """

    debug_log = []
    start_time = time.time()

    def log_debug(message: str, data: Any = None):
        """Helper function to log debug information"""
        timestamp = time.time() - start_time
        log_entry = f"[{timestamp:.2f}s] {message}"
        if data is not None:
            log_entry += f" | Data: {str(data)[:200]}..."
        debug_log.append(log_entry)
        logger.debug(f"🔍 SEARCH: {log_entry}")

    log_debug("=== HYBRID SEARCH DEBUG SESSION STARTED ===")

    try:
        # ====== STEP 1: VALIDATE INPUT ======
        if not text_query or not text_query.strip():
            error_msg = "Text query is empty or None"
            error_result = {
                "error": error_msg,
                "debug_log": debug_log,
                "failure_point": "input_validation",
            }
            # Send RTVI error message
            if event_loop and rtvi_instance:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        send_rtvi_message(rtvi_instance, error_result, text_query),
                        event_loop
                    )
                    logger.info("🚀 RTVI error message scheduled")
                except Exception as rtvi_error:
                    logger.error(f"❌ Error scheduling RTVI error: {rtvi_error}")
            return error_result

        # ====== STEP 2: GENERATE EMBEDDING ======
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_result = {
                "error": "OPENAI_API_KEY not found",
                "debug_log": debug_log,
                "failure_point": "openai_api_key",
            }
            if event_loop and rtvi_instance:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        send_rtvi_message(rtvi_instance, error_result, text_query),
                        event_loop
                    )
                except Exception:
                    pass
            return error_result

        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-large", input=text_query.strip()
            )
            embedding_vector = response.data[0].embedding
            log_debug(f"✅ Embedding generated. Dimensions: {len(embedding_vector)}")
        except Exception as openai_error:
            error_result = {
                "error": f"OpenAI API error: {str(openai_error)}",
                "debug_log": debug_log,
                "failure_point": "openai_api_call",
            }
            if event_loop and rtvi_instance:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        send_rtvi_message(rtvi_instance, error_result, text_query),
                        event_loop
                    )
                except Exception:
                    pass
            return error_result

        # ====== STEP 3: MONGODB CONNECTION & SEARCH ======
        try:
            client = MongoClient(MONGODB_CONNECTION_STRING)
            client.admin.command("ping")
            db = client.real_estate
            collection = db.properties

            # Build filters - Updated for nested structure
            match_conditions = {}
            if min_price is not None or max_price is not None:
                price_filter = {}
                if min_price is not None:
                    price_filter["$gte"] = min_price
                if max_price is not None:
                    price_filter["$lte"] = max_price
                match_conditions["property_details.listed_price"] = price_filter

            if bedrooms is not None:
                match_conditions["property_details.bedrooms"] = bedrooms
            if bathrooms is not None:
                match_conditions["property_details.bathrooms"] = bathrooms
            if property_type is not None:
                match_conditions["property_details.property_type"] = {
                    "$regex": re.escape(property_type),
                    "$options": "i",
                }
            if location_keywords is not None:
                match_conditions["property_details.address"] = {
                    "$regex": re.escape(location_keywords),
                    "$options": "i",
                }
            if mls_genuine is not None:
                match_conditions["property_details.mls_is_genuine"] = mls_genuine

            log_debug(f"Match conditions: {match_conditions}")

            # Build aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": vector_search_index,
                        "path": "embedding",
                        "queryVector": embedding_vector,
                        "numCandidates": min(100, limit * 10),
                        "limit": limit * 5,
                    }
                },
                {"$addFields": {"search_score": {"$meta": "vectorSearchScore"}}},
            ]

            if match_conditions:
                pipeline.append({"$match": match_conditions})

            # Updated projection for nested structure
            pipeline.extend(
                [
                    {
                        "$project": {
                            "_id": 1,
                            "property_url": 1,
                            "property_details.address": 1,
                            "property_details.listed_price": 1,
                            "property_details.currency": 1,
                            "property_details.bedrooms": 1,
                            "property_details.bathrooms": 1,
                            "property_details.property_type": 1,
                            "property_details.mls_description": 1,
                            "property_details.mls_number": 1,
                            "property_details.mls_is_genuine": 1,
                            "processing_info.images_analyzed": 1,
                            "processing_info.status": 1,
                            "ai_analysis_raw": 1,
                            "search_score": 1,
                        }
                    },
                    {"$sort": {"search_score": -1}},
                    {"$limit": limit},
                ]
            )

            log_debug(f"Pipeline stages: {len(pipeline)}")

            # Execute pipeline
            results = list(collection.aggregate(pipeline))
            client.close()
            log_debug(f"Found {len(results)} results")

        except Exception as mongo_error:
            error_result = {
                "error": f"MongoDB error: {str(mongo_error)}",
                "debug_log": debug_log,
                "failure_point": "mongodb_operation",
            }
            if event_loop and rtvi_instance:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        send_rtvi_message(rtvi_instance, error_result, text_query),
                        event_loop
                    )
                except Exception:
                    pass
            return error_result

        # ====== STEP 4: PROCESS RESULTS WITH CORRECT STRUCTURE ======
        formatted_results = []
        similarity_scores = []

        for i, result in enumerate(results):
            search_score = result.get("search_score", 0)
            similarity_scores.append(round(search_score, 4))
            property_details = result.get("property_details", {})
            processing_info = result.get("processing_info", {})

            # Extract image URLs from processing_info.images_analyzed
            image_urls = []
            images_analyzed = processing_info.get("images_analyzed", [])

            if images_analyzed and isinstance(images_analyzed, list):
                # Use the first 3 images from images_analyzed
                image_urls = images_analyzed[:3]
            else:
                # Generate placeholder URLs if no images
                property_id = str(result.get("_id", ""))
                image_urls = [
                    f"https://images.realtor.com/property/{property_id}/main.jpg",
                    f"https://images.realtor.com/property/{property_id}/interior.jpg",
                    f"https://images.realtor.com/property/{property_id}/exterior.jpg",
                ]

            # Format property with correct nested structure access
            formatted_property = {
                "property_id": str(result.get("_id", "")),
                "url": result.get("property_url", ""),
                "image_urls": image_urls,
                "primary_image": image_urls[0] if image_urls else "",
                "address": property_details.get("address", "N/A"),
                "price": property_details.get("listed_price", "N/A"),
                "currency": property_details.get("currency", "CAD"),
                "bedrooms": property_details.get("bedrooms", "N/A"),
                "bathrooms": property_details.get("bathrooms", "N/A"),
                "property_type": property_details.get("property_type", "N/A"),
                "mls_number": property_details.get("mls_number", "N/A"),
                "mls_genuine": property_details.get("mls_is_genuine", "N/A"),
                "search_score": round(search_score, 4),
                "status": processing_info.get("status", "N/A"),
                "description": property_details.get("mls_description", "")[:200] + "..."
                if property_details.get("mls_description", "")
                else "",
            }
            formatted_results.append(formatted_property)

        total_time = time.time() - start_time
        log_debug(f"=== HYBRID SEARCH COMPLETED SUCCESSFULLY in {total_time:.2f}s ===")

        search_result = {
            "search_completed": True,
            "search_type": "hybrid_vector_traditional",
            "query": text_query,
            "embedding_dimensions": len(embedding_vector),
            "filters_applied": {
                "min_price": min_price,
                "max_price": max_price,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "property_type": property_type,
                "location_keywords": location_keywords,
                "mls_genuine": mls_genuine,
            },
            "results_found": len(results),
            "top_similarity_scores": similarity_scores,
            "properties": formatted_results,
            "search_method": "MongoDB vector search + traditional filters",
            "execution_time_seconds": round(total_time, 2),
            "debug_log": debug_log,
            "note": "Hybrid search combining semantic similarity with structured filters",
        }

        # Send RTVI message from within search using run_coroutine_threadsafe
        if event_loop and rtvi_instance:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    send_rtvi_message(rtvi_instance, search_result, text_query),
                    event_loop
                )
                # Don't wait for completion - let it run in background
                logger.info("🚀 RTVI message scheduled from search function")
            except Exception as rtvi_error:
                logger.error(f"❌ Error scheduling RTVI message: {rtvi_error}")

        return search_result

    except Exception as e:
        error_msg = f"Unexpected error in hybrid search: {str(e)}"
        error_result = {
            "error": error_msg,
            "details": str(e),
            "query": text_query,
            "failure_point": "unexpected_error",
            "debug_log": debug_log,
            "full_traceback": traceback.format_exc(),
        }

        # Send RTVI error message from within search
        if event_loop and rtvi_instance:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    send_rtvi_message(rtvi_instance, error_result, text_query),
                    event_loop
                )
                logger.info("🚀 RTVI error message scheduled from search function")
            except Exception as rtvi_error:
                logger.error(f"❌ Error scheduling RTVI error message: {rtvi_error}")

        return error_result


class RealEstateBot:
    """
    Real Estate Bot class with simplified Strands integration following the weather example pattern.
    """

    def __init__(self):
        """Initialize the bot with all necessary components."""
        self.task: Optional[PipelineTask] = None
        self.rtvi: Optional[RTVIProcessor] = None
        self.is_running = False
        self.strands_agent = None

        # Initialize Strands agent for result summarization
        self._initialize_strands_agent()

    def _initialize_strands_agent(self):
        """Initialize the Strands agent with execute_hybrid_search as a tool."""
        self.strands_agent = Agent(
            tools=[execute_hybrid_search],  # Add search as a tool
            system_prompt="""You are a property search specialist. Your role is to:

1. **EXTRACT SEARCH PARAMETERS** from user queries
2. **EXECUTE PROPERTY SEARCHES** using the execute_hybrid_search tool
3. **SUMMARIZE RESULTS** in a conversational format for audio output

When you receive a user query about properties:
- Analyze the query to extract relevant parameters:
  * Location keywords (city, neighborhood, area)
  * Price range (min_price, max_price)
  * Bedrooms and bathrooms (as strings)
  * Property type (house, apartment, condo, etc.)
  * Other preferences

- Call execute_hybrid_search with appropriate parameters
- The search will automatically update the UI with results
- Create a natural, audio-friendly summary including:
  * Number of properties found
  * Key details of top 2-3 properties (address, price, bedrooms/bathrooms)
  * Notable features or recommendations
  * Suggestions for refining the search if needed

Keep responses conversational and suitable for text-to-speech. Avoid special characters.

Example: If user asks "Find me a 3 bedroom house under 500k in Toronto", extract:
- text_query: "3 bedroom house Toronto"
- min_price: None, max_price: 500000
- bedrooms: "3", property_type: "house"
- location_keywords: "Toronto"

Then call the tool and summarize the results naturally.""",
        )

    async def handle_property_search_queries(
        self,
        params: FunctionCallParams,
        query: str,
    ):
        """Handle property search queries with single executor call."""
        logger.info(f"🔍 Handling property search: query='{query}'")

        try:
            # Single executor call - Strands does everything!
            loop = asyncio.get_event_loop()
            
            # Create a bound version of execute_hybrid_search with RTVI parameters
            def execute_search_with_rtvi(*args, **kwargs):
                # Add RTVI parameters to any search call
                return execute_hybrid_search(
                    *args, 
                    event_loop=loop,
                    rtvi_instance=self.rtvi,
                    **kwargs
                )
            
            # Temporarily replace the tool in Strands agent
            original_tool = None
            for i, tool_func in enumerate(self.strands_agent.tools):
                if tool_func.__name__ == 'execute_hybrid_search':
                    original_tool = tool_func
                    self.strands_agent.tools[i] = execute_search_with_rtvi
                    break
            
            # Single executor call - Strands extracts params, calls search, and summarizes
            result = await loop.run_in_executor(None, self.strands_agent, query)
            
            # Restore original tool if needed
            if original_tool:
                for i, tool_func in enumerate(self.strands_agent.tools):
                    if tool_func == execute_search_with_rtvi:
                        self.strands_agent.tools[i] = original_tool
                        break
            
            logger.info("✅ Strands agent completed search and summarization")

            # Return conversational response to LLM
            await params.result_callback(result.message)
            logger.info("🔊 Conversational response sent to LLM")

        except Exception as e:
            logger.error(f"❌ Error in property search handler: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Send error response
            await params.result_callback(
                f"I encountered an error while searching for properties: {str(e)}"
            )

    async def start(self, room_url: str, token: str):
        """Start the bot following the weather example pattern."""
        logger.info("🚀 Starting real estate search bot...")

        # Set running state
        self.is_running = True

        # Initialize services
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # Register the property search handler (like weather example)
        llm.register_direct_function(self.handle_property_search_queries)

        @llm.event_handler("on_function_calls_started")
        async def on_function_calls_started(service, function_calls):
            await tts.queue_frame(
                TTSSpeakFrame("Let me search for that property information.")
            )

        # Setup context and tools
        tools = ToolsSchema(standard_tools=[self.handle_property_search_queries])
        messages = [
            {
                "role": "system",
                "content": """You are a helpful real estate assistant in a WebRTC call. Your goal is to help users find properties.

When users ask about finding properties, call the handle_property_search_queries function with their query.

Your responses should be natural and conversational since they will be converted to audio. Avoid special characters in your answers.

Start by suggesting that users ask about finding properties with specific features or in specific locations.""",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        # Create Daily transport
        transport = DailyTransport(
            room_url,
            token,
            "Real Estate Assistant",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Create RTVI processor and store as instance variable
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]), transport=transport)
        rtvi = self.rtvi

        rtvi_observer = RTVIObserver(
            rtvi,
            params=RTVIObserverParams(
                bot_llm_enabled=True,
                bot_tts_enabled=True,
                user_transcription_enabled=True,
                metrics_enabled=True,
                errors_enabled=True,
            ),
        )
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Build pipeline with RTVI components
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                rtvi,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        # Create and configure task with RTVI observer
        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[rtvi_observer],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            await self.task.queue_frames(
                [context_aggregator.user().get_context_frame()]
            )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            try:
                await transport.capture_participant_transcription(participant["id"])
                logger.info(
                    f"✅ Client connected: {participant.get('info', {}).get('userName', 'Unknown')}"
                )
            except Exception as e:
                logger.error(f"❌ Error in client connected handler: {e}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, participant):
            try:
                logger.info(
                    f"👋 Client disconnected: {participant.get('info', {}).get('userName', 'Unknown')}"
                )
            except Exception as e:
                logger.error(f"❌ Error in client disconnected handler: {e}")

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(self.task)


async def run_bot(room_url: str, token: str):
    """
    Main function that runs the real estate search bot.
    Following the weather example pattern with RTVI integration.
    """
    bot = RealEstateBot()
    try:
        await bot.start(room_url, token)
    except Exception as e:
        logger.error(f"❌ Bot run error: {e}")
