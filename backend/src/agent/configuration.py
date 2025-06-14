import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the French sustainability transcript analyzer."""

    analysis_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for transcript analysis."
        },
    )

    max_segments: int = Field(
        default=50,
        metadata={"description": "The maximum number of segments to process from a transcript."},
    )

    segmentation_temperature: float = Field(
        default=0.3,
        metadata={"description": "Temperature for segmentation tasks (lower = more consistent)."},
    )

    analysis_temperature: float = Field(
        default=0.2,
        metadata={"description": "Temperature for analysis tasks (lower = more consistent)."},
    )

    synthesis_temperature: float = Field(
        default=0.7,
        metadata={"description": "Temperature for synthesis tasks (higher = more creative)."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
