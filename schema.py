from pydantic import BaseModel, Field, field_validator
from config import Config
from typing import List, Optional

config = Config.load_from_env()


# Pydantic models for call response validation
class WebhookResponse(BaseModel):
    id: str
    status: str
    recording_available: bool


class CallRequest(BaseModel):
    phone_number: str = Field(default=config.AGENT_PHONE_NUMBER)
    prompt: str
    webhook_url: str = Field(default=config.WEBHOOK_URL)

    @field_validator("prompt")
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt must not be empty")
        return v


class CallResult(BaseModel):
    id: str
    transcription: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None


class Path(BaseModel):
    steps: List[str]
    call_result: CallResult


class DiscoveredTree(BaseModel):
    paths: List[Path] = Field(default_factory=list)

    def add_path(self, path: Path):
        self.paths.append(path)

    def get_path(self, steps: List[str]) -> Optional[Path]:
        for path in self.paths:
            if path.steps == steps:
                return path
        return None
