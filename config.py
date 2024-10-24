from dotenv import load_dotenv
import os


from pydantic import BaseModel, Field


class Config(BaseModel):
    API_TOKEN: str
    API_URL: str
    OPENAI_API_KEY: str
    WEBHOOK_URL: str
    WEBHOOK_PORT: int = Field(default=8080)
    AGENT_PHONE_NUMBER: str
    CONCURRENT_CALLS: int = Field(default=1)
    MAX_DEPTH: int = Field(default=10)
    TIMEOUT_SECONDS: int = Field(default=300)

    @classmethod
    def load_from_env(cls):
        load_dotenv()
        return cls(  # type: ignore
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),  # type: ignore
            API_TOKEN=os.getenv("API_TOKEN"),  # type: ignore
            API_URL=os.getenv("API_URL"),  # type: ignore
            WEBHOOK_URL=os.getenv("WEBHOOK_URL"),  # type: ignore
            AGENT_PHONE_NUMBER=os.getenv("AGENT_PHONE_NUMBER"),  # type: ignore
            WEBHOOK_PORT=int(os.getenv("WEBHOOK_PORT")),  # type: ignore
            CONCURRENT_CALLS=int(os.getenv("CONCURRENT_CALLS")),  # type: ignore
            MAX_DEPTH=int(os.getenv("MAX_DEPTH")),  # type: ignore
            TIMEOUT_SECONDS=int(os.getenv("TIMEOUT_SECONDS")),  # type: ignore
        )
