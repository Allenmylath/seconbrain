#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import uuid
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

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
            return {
                "error": error_msg,
                "debug_log": debug_log,
                "failure_point": "input_validation",
            }

        # ====== STEP 2: GENERATE EMBEDDING ======
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "error": "OPENAI_API_KEY not found",
                "debug_log": debug_log,
                "failure_point": "openai_api_key",
            }

        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-large", input=text_query.strip()
            )
            embedding_vector = response.data[0].embedding
            log_debug(f"✅ Embedding generated. Dimensions: {len(embedding_vector)}")
        except Exception as openai_error:
            return {
                "error": f"OpenAI API error: {str(openai_error)}",
                "debug_log": debug_log,
                "failure_point": "openai_api_call",
            }

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
            return {
                "error": f"MongoDB error: {str(mongo_error)}",
                "debug_log": debug_log,
                "failure_point": "mongodb_operation",
            }

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

        return {
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

    except Exception as e:
        error_msg = f"Unexpected error in hybrid search: {str(e)}"
        return {
            "error": error_msg,
            "details": str(e),
            "query": text_query,
            "failure_point": "unexpected_error",
            "debug_log": debug_log,
            "full_traceback": traceback.format_exc(),
        }


class RealEstateBot:
    """
    Real Estate Bot class with direct RTVI processing.
    Eliminates queue management to prevent blocking and shutdown issues.
    """

    def __init__(self):
        """Initialize the bot with all necessary components."""
        self.task: Optional[PipelineTask] = None
        self.rtvi: Optional[RTVIProcessor] = None
        self.is_running = False
        self.strands_agent = None

        # Initialize Strands agent for conversational responses
        self._initialize_strands_agent()

    def _initialize_strands_agent(self):
        """Initialize the Strands agent with MongoDB schema knowledge."""
        self.strands_agent = Agent(
            system_prompt="""You are a MongoDB real estate search specialist. Your role is to execute property searches with full knowledge of the database structure and implement progressive search fallback strategies.

## MONGODB SCHEMA KNOWLEDGE:
**Collection:** real_estate.properties

**Document Structure:**
```
{
  "_id": ObjectId,
  "property_url": string,
  "property_details": {
    "address": string,
    "listed_price": number,
    "currency": string,
    "bedrooms": string,
    "bathrooms": string,
    "property_type": string,
    "mls_is_genuine": boolean,
    "description": string
  },
  "embedding": [number array - 3072 dimensions],
  "ai_analysis_raw": object,
  "processing_info": {"status": string},
  "images": [array of image objects],
  "images_analyzed": array
}
```

**Search Parameters You'll Receive:**
- text_query (REQUIRED): For vector similarity search
- min_price, max_price: Filter property_details.listed_price
- bedrooms: Number of bedrooms (string format)
- bathrooms: Number of bathrooms (string format)
- property_type: Type like "house", "apartment", "condo", etc.
- location_keywords: Specific areas, neighborhoods, cities
- mls_genuine: Boolean for MLS verified properties

## SEARCH EXECUTION STRATEGY:

**INITIAL SEARCH:**
- Use ALL provided parameters for precise matching
- Vector search with numCandidates: min(100, limit * 10)
- Apply all filters in $match stage

**IF NO RESULTS (results_found = 0):**
1. **First Fallback:** Remove price constraints (keep other filters)
2. **Second Fallback:** Remove bedrooms/bathrooms constraints  
3. **Third Fallback:** Remove property_type constraint
4. **Final Fallback:** Keep only text_query and location_keywords

**PROGRESSIVE RELAXATION LOGIC:**
- Execute search with current parameters
- If results_found = 0, automatically try next relaxation level
- Track which parameters were relaxed
- Return information about relaxations made

## RESPONSE FORMAT:
Always return structured search results with clear indication of which parameters were used/relaxed, total results found, formatted property data with images, and user-friendly explanation of any parameter relaxations made.

Your goal is to find the best possible property matches while being transparent about search parameter adjustments. Based on the search results provided, create a clear, conversational summary suitable for audio output. Focus on the most relevant details and be encouraging. Don't include special characters.""",
        )

    async def handle_property_search_queries(
        self,
        params: FunctionCallParams,
        text_query: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        bedrooms: Optional[str] = None,
        bathrooms: Optional[str] = None,
        property_type: Optional[str] = None,
        location_keywords: Optional[str] = None,
        mls_genuine: Optional[bool] = None,
        limit: int = 10,
    ):
        """Handle property search with custom ThreadPoolExecutor - no queue."""
        logger.info(
            f"🔍 Handling search: query='{text_query}', bedrooms='{bedrooms}', type='{property_type}'"
        )

        try:
            # Create custom executor with 4 workers
            executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="real-estate-"
            )

            # Execute search in custom executor to avoid blocking
            loop = asyncio.get_running_loop()
            search_data = await loop.run_in_executor(
                executor,  # Custom executor instead of None
                execute_hybrid_search,
                text_query,
                min_price,
                max_price,
                bedrooms,
                bathrooms,
                property_type,
                location_keywords,
                mls_genuine,
                limit,
            )

            logger.info(
                f"✅ Search completed. Results found: {search_data.get('results_found', 0)}"
            )

            # Generate conversational response using Strands
            if (
                search_data.get("search_completed")
                and search_data.get("results_found", 0) > 0
            ):
                properties = search_data.get("properties", [])
                summary_prompt = f"""
                I found {search_data.get('results_found', 0)} properties for the search "{text_query}".
                Here are the top {len(properties)} results:
                
                {chr(10).join([f"Property {i+1}: {prop.get('address', 'Address not available')} - ${prop.get('price', 'Price not listed')} - {prop.get('bedrooms', 'N/A')} bedrooms, {prop.get('bathrooms', 'N/A')} bathrooms" for i, prop in enumerate(properties[:3])])}
                
                Create a friendly, conversational summary of these results.
                """
            else:
                error_msg = search_data.get("error", "No properties found")
                summary_prompt = f'No properties were found for the search "{text_query}". Error: {error_msg}. Provide a helpful response.'

            # Generate conversational response using same custom executor
            logger.info("🤖 Generating conversational response with Strands...")
            result = await loop.run_in_executor(
                executor,
                self.strands_agent,
                summary_prompt,  # Same custom executor
            )

            # Clean up executor when done
            executor.shutdown(wait=False)

            # Send audio response via TTS
            await params.result_callback(result.message)
            logger.info("🔊 Audio response sent")

            # Send RTVI message directly (no queue!)
            await self._send_rtvi_message(search_data, text_query)

        except Exception as e:
            logger.error(f"❌ Error in property search handler: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Send error response
            await params.result_callback(
                f"I encountered an error while searching for properties: {str(e)}"
            )

            # Send error RTVI message
            await self._send_rtvi_error_message(text_query, str(e))

    async def _send_rtvi_message(self, search_data: Dict[str, Any], text_query: str):
        """Send RTVI message directly without queue."""
        try:
            if not self.rtvi:
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

                # Add timeout to prevent RTVI blocking
                try:
                    await asyncio.wait_for(
                        self.rtvi.push_frame(server_message_frame), timeout=5.0
                    )
                    logger.info(
                        f"✅ RTVI message sent directly - {len(search_data['properties'])} properties"
                    )
                except asyncio.TimeoutError:
                    logger.warning("⚠️ RTVI message send timed out")

            else:
                logger.warning("⚠️ No valid search results to send via RTVI")

        except Exception as e:
            logger.error(f"❌ Error sending RTVI message: {e}")
            # Don't re-raise - we don't want RTVI errors to crash the search

    async def _send_rtvi_error_message(self, text_query: str, error_message: str):
        """Send RTVI error message directly."""
        try:
            if not self.rtvi:
                return

            rtvi_message_data = {
                "type": "property_search_error",
                "timestamp": time.time(),
                "search_id": str(uuid.uuid4()),
                "query": text_query,
                "error": error_message,
            }

            server_message_frame = RTVIServerMessageFrame(data=rtvi_message_data)

            # Add timeout for error messages too
            try:
                await asyncio.wait_for(
                    self.rtvi.push_frame(server_message_frame), timeout=3.0
                )
                logger.info("✅ RTVI error message sent directly")
            except asyncio.TimeoutError:
                logger.warning("⚠️ RTVI error message send timed out")

        except Exception as e:
            logger.error(f"❌ Error sending RTVI error message: {e}")

    async def start(self, room_url: str, token: str):
        """Start the bot with direct RTVI processing."""
        logger.info("🚀 Starting real estate search bot with direct RTVI messaging...")

        # Set running state
        self.is_running = True

        # Initialize services
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # Register the bound method as a function
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
                "content": """You are a helpful real estate assistant in a WebRTC call. Your primary role is to:

1. **ANALYZE USER QUERIES** and extract optimal search parameters for property searches
2. **UNDERSTAND CONVERSATION CONTEXT** to infer unstated user preferences  
3. **PROVIDE CONVERSATIONAL RESPONSES** while your output will be converted to audio

## SEARCH PARAMETER EXTRACTION:
When users ask about properties, extract these parameters from their query and conversation history:

**REQUIRED:**
- text_query: The semantic search text (always required)

**OPTIONAL FILTERS:**
- min_price, max_price: Price range in numbers
- bedrooms: Number of bedrooms (string format)
- bathrooms: Number of bathrooms (string format)
- property_type: Type like "house", "apartment", "condo", etc.
- location_keywords: Specific areas, neighborhoods, cities
- mls_genuine: Boolean for MLS verified properties

**CONTEXT ANALYSIS:**
- Track user preferences mentioned earlier in conversation
- Infer missing parameters from context (e.g., if user mentioned budget before)
- Consider family size hints for bedroom/bathroom needs
- Remember location preferences from previous queries

**PROGRESSIVE SEARCH STRATEGY:**
- Start with specific parameters extracted from query
- If no results, prepare fallback parameters (fewer filters)
- Always prioritize user's explicit requirements

## RESPONSE STYLE:
- Conversational and audio-friendly (no special characters)
- Help users refine their search criteria
- Suggest specific features to search for
- When calling search function, pass ALL extracted parameters clearly

Start by suggesting users ask about finding houses with specific features or locations.""",
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
                # transcription_enabled=True,
            ),
        )

        # Create RTVI processor and store as instance variable
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]), transport=transport)
        rtvi = self.rtvi  # Keep local variable for compatibility

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
                rtvi,  # Add RTVI processor early in pipeline
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
            # Kick off the conversation
            await self.task.queue_frames(
                [context_aggregator.user().get_context_frame()]
            )

        # Event handlers with better error handling
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
                # Don't auto-stop on client disconnect - let multiple users connect
                # await self.stop()
            except Exception as e:
                logger.error(f"❌ Error in client disconnected handler: {e}")

            # Run the pipeline

        runner = PipelineRunner()
        await runner.run(self.task)


async def run_bot(room_url: str, token: str):
    """
    Main function that runs the real estate search bot.
    Now uses direct RTVI processing without queue management.
    """
    bot = RealEstateBot()
    try:
        await bot.start(room_url, token)
    except Exception as e:
        logger.error(f"❌ Bot run error: {e}")
