import asyncio
import os
from call_management.call_manager import CallManager
from discovery.agent import DiscoveryAgent
from discovery.output_generator import OutputGenerator
from config import Config
from discovery.phone_tree import PhoneTree
import sys
import aiohttp
from openai import AsyncOpenAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


async def process_recording(
    file_path: str, config: Config, session: aiohttp.ClientSession
) -> dict:
    call_manager = CallManager(config)
    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    transcription = await call_manager.transcribe_audio("test_call", audio_data)
    agent = DiscoveryAgent(call_manager, OutputGenerator(), call_manager.openai)
    options = await agent.phone_tree.extract_path(transcription)

    return {"transcription": transcription, "options": options}


async def test_parsing_and_merging() -> list[dict]:
    config = Config.load_from_env()
    openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    discovery_agent = DiscoveryAgent(
        CallManager(config), OutputGenerator(), openai_client
    )
    recordings_dir = "recordings"
    selected_recordings = [
        "cm2nutthy00x710bcxitp9v74.wav",
        "cm2nv3umv00zxh0llu0tpvn6m.wav",
    ]

    results = []
    async with aiohttp.ClientSession() as session:
        for filename in selected_recordings:
            if filename.endswith(".wav"):
                file_path = os.path.join(recordings_dir, filename)
                result = await process_recording(file_path, config, session)
                results.append(result)
                extracted_path = await discovery_agent.phone_tree.extract_path(
                    result["transcription"]
                )
                await discovery_agent.phone_tree.add_path(extracted_path)

    print("\nFinal Phone Tree Structure:")
    discovery_agent.phone_tree.print_tree()

    return results
