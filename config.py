from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATASET: str

    class Config:
        env_file = './.env'


settings = Settings()
