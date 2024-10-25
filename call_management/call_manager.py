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
from deepgram.utils import verboselogs
import aiofiles


class CallManager:
    """
    Manages calls to the Hamming API and handles webhooks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.openai = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.deepgram_config = DeepgramClientOptions(
            api_key=config.DEEPGRAM_API_KEY,
            verbose=verboselogs.DEBUG,
        )
        self.deepgram = DeepgramClient("", self.deepgram_config)

        self.call_futures: Dict[str, asyncio.Future] = {}
        self.active_call_ids = set()
        self.ended_calls_waiting_recording = set()
        logger.info("CallManager initialized")

    async def make_call(self, phone_number: str, prompt: str) -> CallResult:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                return await asyncio.wait_for(
                    self._make_call_attempt(phone_number, prompt),
                    timeout=self.config.CALL_TIMEOUT_SECONDS * 2,  # Double the timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Call attempt {attempt + 1} timed out")
                return CallResult(
                    id="timeout",
                    status="timeout",
                    message=f"Call timed out after "
                    f"{self.config.CALL_TIMEOUT_SECONDS * 2} seconds",
                )
            except Exception as e:
                logger.error(f"Call attempt {attempt + 1} failed: {str(e)}")

        return CallResult(
            id="error",
            status="failed",
            message=f"All {self.config.MAX_RETRIES} call attempts failed",
        )

    async def _make_call_attempt(self, phone_number: str, prompt: str) -> CallResult:
        """
        Makes a single call attempt to the specified phone number with the given prompt.
        """
        logger.info(f"Initiating call to {phone_number}")
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
                logger.debug(f"Call initiated with ID: {call_id}")

            # Add the future to self.call_futures immediately after getting the call_id
            self.active_call_ids.add(call_id)
            self.call_futures[call_id] = future

            # Wait for the webhook to resolve the future
            try:
                await asyncio.wait_for(
                    self.call_futures[call_id], timeout=self.config.CALL_TIMEOUT_SECONDS
                )
                if call_id in self.ended_calls_waiting_recording:
                    # Wait a bit longer for the recording
                    try:
                        await asyncio.wait_for(
                            self.call_futures[call_id], timeout=30
                        )  # Additional 30 seconds for recording
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout waiting for recording "
                            f"after call end for {call_id}"
                        )
                        return CallResult(
                            id=call_id,
                            status="timeout",
                            message="Timeout waiting for recording after call end",
                        )

                recording = await self.get_recording(call_id)
                transcription = await self.transcribe_audio(call_id, recording)
                return CallResult(
                    id=call_id, status="completed", transcription=transcription
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for call {call_id} to complete")
                return CallResult(
                    id=call_id,
                    status="timeout",
                    message="Timeout waiting for call to complete",
                )
            finally:
                self.active_call_ids.discard(call_id)
                self.call_futures.pop(call_id, None)

        except Exception as e:
            logger.error(f"Unexpected error in make_call: {str(e)}")
            return CallResult(
                id="error",
                status="error",
                message=f"An unexpected error occurred: {str(e)}",
            )

    async def handle_webhook(self, webhook_data: Dict[str, Any]):
        """
        Handles a webhook from the Hamming API.
        """
        logger.info(f"Received webhook: {webhook_data}")

        call_id = webhook_data.get("id")
        status = webhook_data.get("status")

        if not call_id:
            logger.warning("Received webhook with no call ID")
            return

        if call_id not in self.active_call_ids:
            logger.warning(f"Received webhook for unknown call ID: {call_id}")
            return

        if status == "event_phone_call_ended":
            logger.info(f"Call ended for ID: {call_id}")
        elif status == "event_recording":
            logger.info(f"Recording available for call ID: {call_id}")

            if call_id in self.call_futures:
                future = self.call_futures.get(call_id)
                if future:
                    if not future.done():
                        future.set_result(True)
            self.active_call_ids.discard(call_id)
            self.call_futures.pop(str(call_id), None)

    async def get_recording(self, call_id: str) -> bytes:
        """
        Retrieves the recording for a call from the Hamming API.
        """
        logger.info(f"Getting recording for call ID: {call_id}")
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
                    logger.debug(f"Recording retrieved for call {call_id}")
                    return await response.read()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to retrieve recording for call {call_id}: {str(e)}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"Timeout retrieving recording for call {call_id}")
            raise

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
                ).transcribe_file(payload, options, timeout=self.config.TIMEOUT_SECONDS)
                after = datetime.now()

                logger.info(f"Deepgram transcription: {response}")
                logger.info(f"Deepgram transcription took {after - before} seconds")

                # Process transcription
                transcript = await self.process_deepgram_response(response)
                logger.info(f"Transcription: {transcript}")
                return transcript
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {str(e)}")
            raise

    async def process_deepgram_response(self, response: PrerecordedResponse) -> str:
        """
        Process the Deepgram response and format the conversation.
        """
        try:
            channels = response.results.channels
            if len(channels) != 2:
                raise ValueError(f"Expected 2 channels, got {len(channels)}")

            agent_words = channels[0].alternatives[0].words
            customer_words = channels[1].alternatives[0].words

            # Combine words into a single timeline
            all_words = agent_words + customer_words
            all_words.sort(key=lambda w: w.start)

            conversation = []
            current_speaker = None
            current_utterance = []

            for word in all_words:
                speaker = "Agent" if word in agent_words else "Customer"

                if speaker != current_speaker:
                    if current_utterance:
                        conversation.append(
                            f"{current_speaker}: {' '.join(current_utterance)}"
                        )
                        current_utterance = []
                    current_speaker = speaker

                current_utterance.append(word.punctuated_word)

            # Add the last utterance
            if current_utterance:
                conversation.append(f"{current_speaker}: {' '.join(current_utterance)}")

            return "\n".join(conversation)
        except Exception as e:
            logger.error(f"Error in process_deepgram_response: {str(e)}")
            raise

    async def close(self):
        logger.info("Closing CallManager session")
        self.active_call_ids.clear()
        self.ended_calls_waiting_recording.clear()
        for future in self.call_futures.values():
            if not future.done():
                future.cancel()
        await self.session.close()

    async def make_call_with_retry(
        self, phone_number: str, prompt: str, max_retries: int = 3
    ) -> CallResult:
        """
        Makes a call with retry logic, including exponential backoff.
        """
        for attempt in range(max_retries):
            try:
                result = await self._make_call_attempt(phone_number, prompt)
                if result.status == "completed":
                    return result
                logger.warning(f"Call attempt {attempt + 1} failed: {result.message}")
            except Exception as e:
                logger.error(f"Error during call attempt {attempt + 1}: {str(e)}")
            await asyncio.sleep(2**attempt)  # Exponential backoff
        return CallResult(
            id="error",
            status="error",
            message=f"Failed after {max_retries} attempts",
        )
