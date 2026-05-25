import os
import json
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Any, Optional,Dict, List
from collections import defaultdict
from supabase import create_client, Client
from postgrest.exceptions import APIError
from datetime import date
import httpx


# Load env
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dolrgcazdbvnyknstiso.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY", ""))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

from agent import build_graph

# App
app = FastAPI(
    title="Sri Lanka Travel Agent API",
    description="LangGraph-powered Sri Lankan travel assistant.",
    version="2.0.0",
)

origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
cors_origins = [origin.strip() for origin in origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_graph()


# In-memory session store
# session_id → list of {role, content} dicts
# Persists for lifetime of the server process
# Supabase is used for permanent storage across restarts
session_memory: Dict[str, List[Dict[str, str]]] = defaultdict(list)

MAX_HISTORY = 10 

# Request models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None 
    user_email: str

class StoreMessageRequest(BaseModel):
    session_id:       Optional[str] = None
    userquery:     str
    agentresponse: Any
    user_email:   Optional[str] = None 

class SessionHistoryRequest(BaseModel):
    session_id: str
    email:      Optional[str] = None

class CreateSessionRequest(BaseModel):
    session_id:    str
    email:      str
    title:      str = "New Chat"

class GetSessionsRequest(BaseModel):
    email: str

class LoadSessionRequest(BaseModel):
    session_id: str
    email:   str


# Memory Helpers
async def load_history_from_supabase(session_id: str, user_email: str) -> List[Dict[str, str]]:
    """
    On first request for a session, load history from Supabase
    so memory survives server restarts.
    """
    try:
        query = supabase.table("chat_messages") \
            .select("role, content") \
            .eq("session_id", session_id )
            
        if user_email:
            query = query.eq("useremail", user_email)
            
        res = query \
            .order("created_at", desc=False) \
            .limit(MAX_HISTORY * 2) \
            .execute()
            
        
        if not res.data:
            return []

        # Cleanly map the results
        print(f"DATA",[
              {"role": r["role"], "content": r["content"].get("result") if isinstance(r["content"], dict) else r["content"]} 
            for r in res.data
        ])
        
        return [
            {"role": r["role"], "content": r["content"].get("result") if isinstance(r["content"], dict) else r["content"]} 
            for r in res.data
        ]
        
        # if res.data:
        #     # print([{"role": r["role"], "content": r["content"]} for r in res.data])
        #     return [{"role": r["role"], "content": r["content"]} for r in res.data]
        # return []
    except Exception as e:
        print(f"⚠️ Could not load history: {e}")
        return []
    
    

def get_history_as_messages(session_id: str) -> List:
    """Convert stored history to LangChain message objects."""
    history = session_memory.get(session_id, [])
        
    messages = []
    for h in history[-MAX_HISTORY * 2:]:   # keep last N exchanges
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            messages.append(AIMessage(content=h["content"]))
    return messages



def save_to_memory(session_id: str, user_query: str, ai_response: str):
    """Save exchange to in-memory store."""
    session_memory[session_id].append({"role": "user",      "content": user_query})
    session_memory[session_id].append({"role": "assistant", "content": ai_response})

    # Trim to MAX_HISTORY pairs
    if len(session_memory[session_id]) > MAX_HISTORY * 2:
        session_memory[session_id] = session_memory[session_id][-(MAX_HISTORY * 2):]



# Endpoints
@app.post("/chat")
async def chat_with_agent(request: QueryRequest):
    try:
        session_id = request.session_id or "default"
        email = request.user_email
        print(f"📝 Session: {session_id}")

        # Load history from Supabase if not in memory yet
        if session_id not in session_memory:
            print(f"🔄 Loading history from Supabase for: {session_id}")
            session_memory[session_id] = await load_history_from_supabase(
                session_id, 
                user_email=email
            )
   

        # Build message history for agent 
        history_messages = get_history_as_messages(session_id)
        print(f"📚 History length: {len(history_messages)} messages")
        
        final_state = graph_app.invoke({
            "messages": history_messages +  [HumanMessage(content=request.query)],
            "session_id": session_id,
        })

        weather = final_state.get("weather",None)
        # if isinstance(weather, dict):
        #     weather_out = weather.get("summary") or weather
        # else:
        #     weather_out = weather
        weather_out = weather.get("summary") if isinstance(weather, dict) else weather

        hotels = final_state.get("hotels")
        
        result = final_state.get("result", "")

        save_to_memory(session_id, request.query, result or "")
        
        return {
            "success": True,
            'session_id': session_id,
            "destination": final_state.get("destination") or final_state.get("destination_place", ""),
            "latitude": final_state.get("latitude"),
            "longitude": final_state.get("longitude"),
            "result": final_state.get("result", ""),
            "weather": final_state.get("weather",None),
            "hotels": hotels,
            "web_description": final_state.get("web_description"),
            "Bus_info": final_state.get("Bus_info"),
            "Train_info": final_state.get("Train_info"),
            "emergency": final_state.get("emergency"),
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "result":  "An error occurred while processing your request.",
            "error":   str(e),
        }



@app.get("/token_status")
async def token_status():
    """
    Sends a minimal request to Groq and reads the rate-limit headers
    from the response — no manual tracking needed.
    Returns remaining tokens, limit, and reset time directly from Groq.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      os.getenv("GROQ_MODEL", ''),
                "messages":   [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=10,
        )
 
    h = response.headers
    return {
        "success": True,
        "tokens": {
            "limit":     h.get("x-ratelimit-limit-tokens"),
            "remaining": h.get("x-ratelimit-remaining-tokens"),
            "resets_in": h.get("x-ratelimit-reset-tokens"),  
        },
        "requests": {
            "limit":     h.get("x-ratelimit-limit-requests"),
            "remaining": h.get("x-ratelimit-remaining-requests"),
            "resets_in": h.get("x-ratelimit-reset-requests"),
        },
    }
 
 
 
 

@app.post("/store_message")
async def store_message(request: StoreMessageRequest):
    try:
        print("--- Store Message ---")
        
        email = request.user_email if request.user_email else None
        
        supabase.table("chat_messages").insert({
            "session_id": request.session_id,
            "role":    "user",
            "content": request.userquery,
            "useremail": email,
        }).execute()
        
        
        if isinstance(request.agentresponse, dict):
            content = (
                request.agentresponse.get("result") or
                request.agentresponse.get("message") or
                request.agentresponse.get("web_description") or
                str(request.agentresponse)
            )
        else:
            content = str(request.agentresponse)
            

        supabase.table("chat_messages").insert({
            "session_id": request.session_id,
            "role":    "assistant",
            "content": request.agentresponse,
            # "content": str(request.agentresponse),
            "useremail": email,
        }).execute()
        
        # Auto create session with first message as title
        title = request.userquery[:60]
        
        if request.session_id and email != "unknown":
            title = request.userquery[:60]
            supabase.table("chat_sessions").upsert({
                "session_id":    request.session_id,
                "email":      email,
                "title":      title,
                "updated_at": "now()",
            }, on_conflict="session_id").execute()
        
        print("--- Store Message Done ---")
        return {"success": True, "message": "Messages stored."}

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



@app.post("/session_history_chat")
async def session_history_chat(request: SessionHistoryRequest):
    try:
        query = supabase.table("chat_messages") \
            .select("role, content, created_at") \
            .eq("session_id", request.session_id)

        if request.email:
            query = query.eq("useremail", request.email)

        res = query \
            .order("created_at", desc=False) \
            .execute()

        return {
            "success":  True,
            "messages": res.data or [],
            "count":    len(res.data or []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/get_sessions")
async def get_sessions(request: GetSessionsRequest):
    try:
        res = supabase.table("chat_sessions") \
            .select("session_id, title, created_at, updated_at") \
            .eq("email", request.email) \
            .order("updated_at", desc=True) \
            .limit(50) \
            .execute()
        return {"success": True, "sessions": res.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/create_session")
async def create_session(request: CreateSessionRequest):
    try:
        # Upsert create if not exists, update title if exists
        supabase.table("chat_sessions").upsert({
            "session_id":    request.session_id,
            "email":      request.email,
            "title":      request.title[:60],  # truncate long titles
            "updated_at": "now()",
        }, on_conflict="session_id").execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/load_session")
async def load_session(request: LoadSessionRequest):
    try:
        res = supabase.table("chat_messages") \
            .select("role, content, created_at") \
            .eq("session_id",  request.session_id) \
            .eq("email",    request.email) \
            .order("created_at", desc=False) \
            .execute()
        return {"success": True, "messages": res.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    try:
        supabase.table("chat_sessions") \
            .delete().eq("session_id", session_id).execute()
        supabase.table("chat_messages") \
            .delete().eq("session_id", session_id).execute()
        if session_id in session_memory:
            del session_memory[session_id]
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/session/{session_id}")
async def clear_session_memory(session_id: str):
    if session_id in session_memory:
        del session_memory[session_id]
    return {"success": True, "message": "Memory cleared"}



@app.get("/health")
async def health():
    return {"status": "ok", "agent": "Sri Lanka Travel Agent v2"}
