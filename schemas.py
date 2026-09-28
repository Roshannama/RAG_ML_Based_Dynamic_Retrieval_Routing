from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
class MessageCreate(BaseModel):
    role: str
    content: str
class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    class Config:
        from_attributes = True
class SessionCreate(BaseModel):
    name: Optional[str] = "New Session"
class SessionOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    messages: List[MessageOut] = []
    class Config:
        from_attributes = True
