from fastapi import APIRouter, Depends, Request, HTTPException
from app.schemas import ChatRequest, ChatResponse, Source, SessionInfo, MessageHistory
from app.auth import verify_api_key
from app.services.session_manager import session_manager
from app.services.groq_service import groq_service
from app.config import settings
import logging
import httpx

logger = logging.getLogger(__name__)
router = APIRouter()

BACKOFFICE_URL = "https://acl-backoffice-api.onrender.com"


async def log_to_backoffice(data: dict):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKOFFICE_URL}/api/analytics/log/",
                json=data,
                timeout=5.0
            )
    except Exception as e:
        logger.warning(f"⚠️ Log back-office échoué: {e}")


@router.post("/chat", response_model=ChatResponse, summary="Envoyer un message au chatbot")
async def chat(
    request: Request,
    body: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    session = session_manager.get_or_create_session(body.session_id)
    session_manager.add_message(session.session_id, "user", body.message)

    rag_service = request.app.state.rag_service
    rag_results = rag_service.search(
        query=body.message,
        top_k=settings.TOP_K_RESULTS,
        country_filter=body.country_filter
    )
    context = rag_service.format_context(rag_results)
    history = session_manager.get_history(session.session_id)

    response_text, tokens_used = groq_service.generate_response(
        user_message=body.message,
        context=context,
        history=history,
        language=body.language.value
    )

    session_manager.add_message(session.session_id, "assistant", response_text)

    seen_sources = set()
    unique_sources = []
    for chunk, score in rag_results:
        if chunk.source_file not in seen_sources:
            seen_sources.add(chunk.source_file)
            unique_sources.append(
                Source(
                    content=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                    source_file=chunk.source_file,
                    relevance_score=round(score, 3)
                )
            )

    logger.info(f"💬 Session {session.session_id[:8]}... | {len(unique_sources)} sources uniques | {tokens_used} tokens")

    await log_to_backoffice({
        "session_id": session.session_id,
        "api_key_used": api_key,
        "user_message": body.message,
        "bot_response": response_text,
        "language": body.language.value,
        "country_filter": body.country_filter,
        "tokens_used": tokens_used or 0,
        "sources_count": len(unique_sources),
    })

    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        language=body.language,
        sources=unique_sources,
        tokens_used=tokens_used,
        model=settings.GROQ_MODEL
    )


@router.get("/session/{session_id}", response_model=SessionInfo, summary="Historique d'une session")
async def get_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    session = session_manager.get_session_info(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' non trouvée")

    return SessionInfo(
        session_id=session.session_id,
        message_count=session.total_messages,
        created_at=session.created_at,
        last_activity=session.last_activity,
        history=[
            MessageHistory(role=msg.role, content=msg.content, timestamp=msg.timestamp)
            for msg in session.messages
        ]
    )


@router.delete("/session/{session_id}", summary="Supprimer une session")
async def delete_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' non trouvée")
    return {"message": f"Session '{session_id}' supprimée avec succès"}