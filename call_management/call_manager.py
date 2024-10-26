import aiohttp
import asyncio
from loguru import logger
from schema import CallRequest, CallResult
from config import Config
import openai
from typing import Dict, Any
import os
from deepgram import (
    PrerecordedOptions,
    FileSource,
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedResponse,
)
from datetime import datetime
import aiofiles



class CallManager:
    """
    Manages calls to the Hamming API and handles webhooks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.openai = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.deepgram_config = DeepgramClientOptions(api_key=config.DEEPGRAM_API_KEY)
        self.deepgram = DeepgramClient("", self.deepgram_config)

        self.call_futures: Dict[str, asyncio.Future] = {}
        self.active_call_ids = set()
        self.ended_calls_waiting_recording = set()
        logger.info("CallManager initialized")

    async def make_call(self, phone_number: str, prompt: str) -> CallResult:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                logger.debug(
                    "Call attempt {} of {}", attempt + 1, self.config.MAX_RETRIES
                )
                return await asyncio.wait_for(
                    self._make_call_attempt(phone_number, prompt),
                    timeout=self.config.CALL_TIMEOUT_SECONDS * 2,  # Double the timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Call attempt {} timed out after {} seconds",
                    attempt + 1,
                    self.config.CALL_TIMEOUT_SECONDS * 2,
                )
                return CallResult(
                    id="timeout",
                    status="timeout",
                    message=f"Call timed out after {self.config.CALL_TIMEOUT_SECONDS * 2} seconds",
                )
            except Exception as e:
                logger.error(
                    "Call attempt {} failed: {}", attempt + 1, str(e), exc_info=True
                )

        logger.error("All {} call attempts failed", self.config.MAX_RETRIES)
        return CallResult(
            id="error",
            status="failed",
            message=f"All {self.config.MAX_RETRIES} call attempts failed",
        )

    async def _make_call_attempt(self, phone_number: str, prompt: str) -> CallResult:
        """
        Makes a single call attempt to the specified phone number with the given prompt.
        """
        logger.info("Initiating call to {}", phone_number)
        try:
            call_request = CallRequest(
                phone_number=phone_number,
                prompt=prompt,
                webhook_url=self.config.WEBHOOK_URL,
            )

            # Create a future for this call before making the API request
            future = asyncio.Future()

            async with self.session.post(
                f"{self.config.API_URL}/exercise/start-call",
                json=call_request.model_dump(),
                headers={"Authorization": f"Bearer {self.config.API_TOKEN}"},
            ) as response:
                response.raise_for_status()
                call_data = await response.json()
                call_id = call_data["id"]
                logger.info("Call initiated with ID: {}", call_id)

            # Add the future to self.call_futures immediately after getting the call_id
            self.active_call_ids.add(call_id)
            self.call_futures[call_id] = future

            # Wait for the webhook to resolve the future
            try:
                logger.debug("Waiting for call {} to complete", call_id)
                await asyncio.wait_for(
                    self.call_futures[call_id], timeout=self.config.CALL_TIMEOUT_SECONDS
                )
                if call_id in self.ended_calls_waiting_recording:
                    # Wait a bit longer for the recording
                    try:
                        logger.debug("Waiting for recording for call {}", call_id)
                        await asyncio.wait_for(
                            self.call_futures[call_id], timeout=30
                        )  # Additional 30 seconds for recording
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Timeout waiting for recording after call end for {}",
                            call_id,
                        )
                        return CallResult(
                            id=call_id,
                            status="timeout",
                            message="Timeout waiting for recording after call end",
                        )

                recording = await self.get_recording(call_id)
                transcription = await self.transcribe_audio(call_id, recording)
                logger.info("Call {} completed successfully", call_id)
                return CallResult(
                    id=call_id, status="completed", transcription=transcription
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for call {} to complete", call_id)
                return CallResult(
                    id=call_id,
                    status="timeout",
                    message="Timeout waiting for call to complete",
                )
            finally:
                self.active_call_ids.discard(call_id)
                self.call_futures.pop(call_id, None)
                logger.debug("Cleaned up resources for call {}", call_id)

        except Exception as e:
            logger.exception("Unexpected error in make_call: {}", str(e))
            return CallResult(
                id="error",
                status="error",
                message=f"An unexpected error occurred: {str(e)}",
            )

    async def handle_webhook(self, webhook_data: Dict[str, Any]):
        """
        Handles a webhook from the Hamming API.
        """
        logger.info("Received webhook: {}", webhook_data)

        call_id = webhook_data.get("id")
        status = webhook_data.get("status")

        if not call_id:
            logger.warning("Received webhook with no call ID")
            return

        if call_id not in self.active_call_ids:
            logger.warning("Received webhook for unknown call ID: {}", call_id)
            return

        if status == "event_phone_call_ended":
            logger.info("Call ended for ID: {}", call_id)
        elif status == "event_recording":
            logger.info("Recording available for call ID: {}", call_id)

            if call_id in self.call_futures:
                future = self.call_futures.get(call_id)
                if future:
                    if not future.done():
                        future.set_result(True)
                        logger.debug("Set future result for call {}", call_id)
            self.active_call_ids.discard(call_id)
            self.call_futures.pop(str(call_id), None)
            logger.debug("Cleaned up resources for completed call {}", call_id)

    async def get_recording(self, call_id: str) -> bytes:
        """
        Retrieves the recording for a call from the Hamming API.
        """
        logger.info("Getting recording for call ID: {}", call_id)
        url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(
                        total=self.config.TIMEOUT_SECONDS,
                        connect=10,
                    ),
                ) as response:
                    response.raise_for_status()
                    logger.debug("Recording retrieved for call {}", call_id)
                    response = await response.read()
                    await session.close()
                    return response
        except aiohttp.ClientError as e:
            logger.error(
                "Failed to retrieve recording for call {}: {}", call_id, str(e)
            )
            raise
        except asyncio.TimeoutError:
            logger.error("Timeout retrieving recording for call {}", call_id)
            raise

    async def process_deepgram_response(self, response: PrerecordedResponse) -> str:
        if not response.results or not response.results.channels:
            logger.warning("No results found in Deepgram response")
            return ""

        transcription = []
        for channel_idx, channel in enumerate(response.results.channels):
            if channel.alternatives:
                speaker = "Agent" if channel_idx == 0 else "Customer"
                for word in channel.alternatives[0].words:
                    transcription.append((speaker, word.word, word.start))

        # Sort the words by start time
        transcription.sort(key=lambda x: x[2])

        # Combine words into sentences
        formatted_transcription = []
        current_speaker = None
        current_sentence = []
        for speaker, word, _ in transcription:
            if speaker != current_speaker:
                if current_sentence:
                    formatted_transcription.append(
                        f"{current_speaker}: {' '.join(current_sentence)}"
                    )
                    current_sentence = []
                current_speaker = speaker
            current_sentence.append(word)

        if current_sentence:
            formatted_transcription.append(
                f"{current_speaker}: {' '.join(current_sentence)}"
            )

        return "\n".join(formatted_transcription)

    async def close(self):
        logger.info("Closing CallManager session")
        self.active_call_ids.clear()
        self.ended_calls_waiting_recording.clear()
        for future in self.call_futures.values():
            if not future.done():
                future.cancel()
        await self.session.close()
        logger.debug("CallManager session closed and resources cleaned up")

    async def transcribe_audio(self, call_id: str, audio_data: bytes) -> str:
        """
        Transcribes audio data using Deepgram and saves the audio file.
        """
        try:
            # Save the audio data to a file
            recordings_folder = "recordings"
            os.makedirs(recordings_folder, exist_ok=True)
            audio_file_path = os.path.join(recordings_folder, f"{call_id}.wav")
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_data)
            logger.debug("Audio file saved: {}", audio_file_path)

            # Read the audio file
            async with aiofiles.open(audio_file_path, "rb") as audio_file:
                buffer_data = await audio_file.read()
                payload: FileSource = {"buffer": buffer_data}
                options: PrerecordedOptions = PrerecordedOptions(
                    model="nova-2-phonecall",
                    smart_format=True,
                    language="en",
                    punctuate=True,
                    multichannel=True,
                )

                # Transcribe using Deepgram
                before = datetime.now()
                response: PrerecordedResponse = await self.deepgram.listen.asyncrest.v(
                    "1"
                ).transcribe_file(
                    payload,
                    options,
                    timeout=self.config.TIMEOUT_SECONDS,
                )
                after = datetime.now()

                logger.info(
                    "Deepgram transcription took {} seconds",
                    (after - before).total_seconds(),
                )

                # Process transcription
                transcription = await self.process_deepgram_response(response)
                logger.debug(f"Transcription for call {call_id}: {transcription}")
                if not transcription.strip():
                    logger.warning(f"Empty transcription for call {call_id}")
                return transcription
        except Exception as e:
            logger.exception(
                "Error in transcribe_audio for call {}: {}", call_id, str(e)
            )
            raise
