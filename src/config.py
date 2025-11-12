from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import yaml, os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    use_finbert: bool = True
    half_life_hours: int = 36
    window_days: int = 10
    weights: dict = Field(default_factory=lambda: {"recency":0.45,"events":0.25,"breadth":0.20,"volume":0.10})
    event_weights: dict = Field(default_factory=dict)
    source_weights: dict = Field(default_factory=dict)
    database_url: str | None = os.getenv("DATABASE_URL")

def load_settings(path: str = "config.yml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return Settings(**y)

def load_stocks(path: str = "data/stocks.yml") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return y.get("stocks", [])
