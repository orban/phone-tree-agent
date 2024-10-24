import aiohttp
import asyncio
from loguru import logger
from schema import CallRequest, CallResult
from config import Config
import openai
import io


class CallManager:
    """
    Manages calls to the Hamming API and handles webhooks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.openai = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.call_futures = {}
        logger.info("CallManager initialized")

    async def make_call(self, phone_number: str, prompt: str) -> CallResult:
        return await self.make_call_with_retry(phone_number, prompt)

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
            self.call_futures[call_id] = future

            # Wait for the webhook to resolve the future
            try:
                final_status = await asyncio.wait_for(
                    future,
                    timeout=self.config.TIMEOUT_SECONDS,
                )

                if final_status == "event_recording":
                    recording = await self.get_recording(call_id)
                    transcription = await self.transcribe_audio(recording)
                    return CallResult(
                        id=call_id,
                        status="completed",
                        transcription=transcription,
                    )
                else:
                    logger.warning(
                        f"Unexpected final status for call {call_id}: {final_status}"
                    )
                    return CallResult(
                        id=call_id,
                        status="failed",
                        transcription="",
                        message=f"Call ended with unexpected status: {final_status}",
                    )

            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for recording on call {call_id}")
                return CallResult(
                    id=call_id,
                    status="timeout",
                    message="Timeout waiting for recording",
                )
            finally:
                # Clean up the future
                self.call_futures.pop(call_id, None)

        except Exception as e:
            logger.error(f"Unexpected error in make_call: {str(e)}")
            return CallResult(
                id="error",
                status="error",
                message=f"An unexpected error occurred: {str(e)}",
            )

    async def handle_webhook(self, webhook_data: dict):
        """
        Handles a webhook from the Hamming API.
        """
        logger.info(f"Received webhook: {webhook_data}")

        call_id = webhook_data.get("id")
        if not call_id:
            logger.error(f"Received webhook without call ID: {webhook_data}")
            return

        if call_id in self.call_futures:
            status = webhook_data.get("status")
            recording_available = webhook_data.get("recording_available", False)

            if status == "event_recording" and recording_available:
                self.call_futures[call_id].set_result(status)
            elif status == "event_phone_call_ended":
                if recording_available:
                    self.call_futures[call_id].set_result("event_recording")
                else:
                    # Wait for the recording to become available
                    logger.info(f"Call {call_id} ended, waiting for recording")
            else:
                logger.debug(
                    f"Received non-terminal webhook for call {call_id}: {status}"
                )
        else:
            logger.warning(f"Received webhook for unknown call ID: {call_id}")

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
                    timeout=aiohttp.ClientTimeout(total=self.config.TIMEOUT_SECONDS),
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

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribes audio data using OpenAI's Whisper model.
        """
        try:
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"

            transcript = await self.openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                timeout=self.config.TIMEOUT_SECONDS,
                temperature=0.0,
            )
            logger.debug(f"Transcription: {transcript.text}")
            return transcript.text
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""

    async def close(self):
        logger.info("Closing CallManager session")
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
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return CallResult(
            id="error",
            status="error",
            message=f"Failed after {max_retries} attempts",
        )
