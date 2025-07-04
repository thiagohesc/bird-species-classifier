from pydantic import BaseModel


class DownloadRequest(BaseModel):
    code: str


class ClearRequest(BaseModel):
    code: str


class PreparerRequest(BaseModel):
    code: str