from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import engine, get_db, SessionLocal
import models
from rag import ask_question
app = FastAPI()
models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def get_or_create_default_session(db: Session):
    sessions = (
        db.query(models.ChatSession)
        .order_by(models.ChatSession.created_at.asc())
        .all()
    )
    if not sessions:
        default = models.ChatSession(name="Session 1")
        db.add(default)
        db.commit()
        db.refresh(default)
        sessions = [default]
    return sessions
@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    session_id: int = None,
    db: Session = Depends(get_db_session),
):
    sessions = get_or_create_default_session(db)
    if session_id is None:
        session_id = sessions[-1].id
    current = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    if not current:
        return RedirectResponse(url="/", status_code=302)
    messages = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.created_at.asc())
        .all()
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "messages": messages,
            "sessions": sessions,
            "current_session_id": session_id,
        },
    )
@app.post("/", response_class=HTMLResponse)
async def chat(
    request: Request,
    question: str = Form(...),
    session_id: int = Form(...),
    db: Session = Depends(get_db_session),
):
    result = ask_question(
        f"session_{session_id}",
        question
    )
    answer = result["answer"]
    user_message = models.ChatMessage(
        session_id=session_id,
        role="user",
        content=question
    )
    db.add(user_message)
    db.commit()
    experience = models.Experience(
        session_id=session_id,
        query=result["question"],
        query_features=result.get(
            "query_features"
        ),
        selected_pipeline=result.get(
            "pipeline"
        ),
        retrieved_documents=result.get(
            "contexts"
        ),
        answer=result["answer"],
        latency=result.get(
            "latency"
        )
    )
    db.add(experience)
    db.commit()
    db.refresh(experience)
    assistant_message = models.ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer,
        experience_id=experience.id
    )
    db.add(assistant_message)
    db.commit()
    sessions = get_or_create_default_session(db)
    messages = (
        db.query(models.ChatMessage)
        .filter(
            models.ChatMessage.session_id == session_id
        )
        .order_by(
            models.ChatMessage.created_at.asc()
        )
        .all()
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "messages": messages,
            "sessions": sessions,
            "current_session_id": session_id,
        },
    )
@app.post("/feedback")
async def submit_feedback(
    experience_id: int = Form(...),
    feedback: str = Form(...),
    feedback_text: str = Form(""),
    db: Session = Depends(get_db_session),
):
    experience = (
        db.query(models.Experience)
        .filter(
            models.Experience.id == experience_id
        )
        .first()
    )
    if not experience:
        return {
            "success": False,
            "message": "Experience not found"
        }
    if feedback not in [
        "positive",
        "negative"
    ]:
        return {
            "success": False,
            "message": "Invalid feedback"
        }
    experience.feedback = feedback
    experience.feedback_text = feedback_text
    db.commit()
    return {
        "success": True,
        "experience_id": experience_id
    }
@app.post("/sessions/new")
async def new_session(db: Session = Depends(get_db_session)):
    count = db.query(models.ChatSession).count()
    session = models.ChatSession(name=f"Session {count + 1}")
    db.add(session)
    db.commit()
    db.refresh(session)
    return RedirectResponse(url=f"/?session_id={session.id}", status_code=303)
@app.post("/sessions/{session_id}/delete")
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db_session),
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    if session:
        db.delete(session)
        db.commit()
    return RedirectResponse(url="/", status_code=303)
