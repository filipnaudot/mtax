from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Relation(ImmutableModel):
    source: str
    target: str
    kind: Literal["attack", "support"]


class Argument(ImmutableModel):
    label: str
    text: str


class Disclosure(ImmutableModel):
    arguments: tuple[Argument, ...] = ()
    relations: tuple[Relation, ...] = Field(min_length=1)


class Pass(ImmutableModel):
    action: Literal["pass"]
    reason: str | None = None
