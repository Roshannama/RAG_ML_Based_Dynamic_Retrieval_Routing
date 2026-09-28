from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )
    name = Column(
        String,
        default="New Session"
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    experiences = relationship(
        "Experience",
        back_populates="session",
        cascade="all, delete-orphan"
    )
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )
    session_id = Column(
        Integer,
        ForeignKey("chat_sessions.id")
    )
    role = Column(
        String
    )
    content = Column(
        Text
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
    experience_id = Column(
        Integer,
        ForeignKey("experiences.id"),
        nullable=True
    )
    session = relationship(
        "ChatSession",
        back_populates="messages"
    )
    experience = relationship(
        "Experience",
        back_populates="message"
    )
class Experience(Base):
    __tablename__ = "experiences"
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )
    session_id = Column(
        Integer,
        ForeignKey("chat_sessions.id"),
        nullable=False
    )
    query = Column(
        Text,
        nullable=False
    )
    query_features = Column(
        JSON,
        nullable=True
    )
    selected_pipeline = Column(
        String,
        nullable=False
    )
    router_confidence = Column(
        Float,
        nullable=True
    )
    retrieved_documents = Column(
        JSON,
        nullable=True
    )
    answer = Column(
        Text,
        nullable=False
    )
    feedback = Column(
        String,
        nullable=True
    )
    feedback_text = Column(
        Text,
        nullable=True
    )
    latency = Column(
        Float,
        nullable=True
    )
    faithfulness = Column(
        Float,
        nullable=True
    )
    context_relevance = Column(
        Float,
        nullable=True
    )
    answer_relevance = Column(
        Float,
        nullable=True
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
    session = relationship(
        "ChatSession",
        back_populates="experiences"
    )
    message = relationship(
        "ChatMessage",
        back_populates="experience"
    )
