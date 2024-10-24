from aiohttp import web
from loguru import logger
from call_management.call_manager import CallManager


class WebhookServer:
    """
    Manages webhooks from the Hamming API.
    """

    def __init__(self, call_manager: CallManager, port=8080):
        self.app = web.Application()
        self.app.router.add_post("/webhook", self.handle_webhook)
        self.port = port
        self.call_manager = call_manager
        self.webhook_listeners = []

    async def handle_webhook(self, request):
        """
        Handles webhooks from the Hamming API.

        Args:
            request (aiohttp.web.Request): The request object.
        """
        data = await request.json()
        logger.info(f"Received webhook: {data}")
        await self.call_manager.handle_webhook(data)
        for listener in self.webhook_listeners:
            await listener(data)
        return web.Response(text="Webhook received")

    def add_webhook_listener(self, listener):
        """
        Adds a listener to the webhook server.

        Args:
            listener (Callable): A callable that takes a dictionary as an argument.
        """
        self.webhook_listeners.append(listener)

    async def start(self):
        """
        Starts the webhook server.
        """
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.site = web.TCPSite(runner, "localhost", self.port)
        await self.site.start()
        logger.info(f"Webhook server started on port {self.port}")

    async def stop(self):
        """
        Stops the webhook server.
        """
        await self.site.stop()
        logger.info("Webhook server stopped")
