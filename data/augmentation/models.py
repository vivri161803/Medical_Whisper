from pydantic import BaseModel, FilePath, Field, AliasChoices
from typing import List, Optional

class AudioTextPair(BaseModel):
    id: str
    audio_path: FilePath = Field(..., validation_alias=AliasChoices("audio_path", "audio_filepath"))
    transcript: str = Field(..., validation_alias=AliasChoices("transcript", "text"))
    duration_sec: Optional[float] = Field(None, le=30.0, description="Duration must be <= 30.0 seconds")

class AugmentedAudioTextPair(AudioTextPair):
    augmented_audio_path: FilePath
    applied_augmentations: List[str]
    duration_sec: float = Field(..., le=30.0)

