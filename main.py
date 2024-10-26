import asyncio
from loguru import logger
import sys
from config import Config
from discovery.agent import DiscoveryAgent
from call_management.call_manager import CallManager
from discovery.output_generator import OutputGenerator
from call_management.webhook_server import WebhookServer
from openai import AsyncOpenAI
from discovery.phone_tree import PhoneTree

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")
logger.add("app.log", rotation="1 MB", level="DEBUG", format="{time} {level} {message}")


async def main():
    """
    Main function to run the discovery process.
    """
    config = Config.load_from_env()
    call_manager = None
    webhook_server = None
    try:
        call_manager = CallManager(config)
        output_generator = OutputGenerator()
        openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        discovery_agent = DiscoveryAgent(call_manager, output_generator, openai_client)
        discovery_agent.phone_tree = PhoneTree(openai_client, output_generator)

        webhook_server = WebhookServer(call_manager, config.WEBHOOK_PORT)
        await webhook_server.start()

        results = await discovery_agent.explore_phone_tree(
            config.AGENT_PHONE_NUMBER,
        )

        if results:
            logger.info(f"Exploration results: {results}")
            output_generator.generate_summary_report(discovery_agent.phone_tree)
            await output_generator.print_tree(discovery_agent.phone_tree.root)
            mermaid_graph = await output_generator.generate_mermaid_graph(
                discovery_agent.phone_tree
            )
            with open("phone_tree.mmd", "w") as f:
                f.write(mermaid_graph)
            print("Exploration complete. Mermaid graph saved to phone_tree.mmd")
        else:
            logger.error("No results found from exploration")
    except asyncio.TimeoutError:
        logger.error("Exploration timed out")
    except Exception as e:
        logger.exception(f"An error occurred during exploration: {str(e)}")
    finally:
        if call_manager:
            await call_manager.close()
        if webhook_server:
            await webhook_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
