import os
import json
import traceback
from typing import TypedDict, Dict, Any, Literal, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph, END
from pydantic import Field, BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from tools import (
    EmergencyData,
    GetTrainScheduleTool,
    GetHotelSearchTool,
    GetWeatherOfDestination,
    WebSearchDestinationTool,
    GetGeocodingOfLocation,
    GetBusRouteTool
)

# Load env
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")
RAPIDAPI_KEY     = os.getenv("RAPIDAPI_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME       = os.getenv("INDEX_NAME", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", '')

os.environ["GROQ_API_KEY"]   = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["RAPIDAPI_KEY"]   = RAPIDAPI_KEY

# External clients
embedder      = SentenceTransformer("all-MiniLM-L6-v2")
groq_client   = Groq(api_key=GROQ_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index         = pinecone_client.Index(INDEX_NAME)


# States
class AgentState(TypedDict, total=False):
    messages:         List[BaseMessage]
    session_id:       Optional[str]  
    destination:      Optional[str]    
    destination_place:Optional[str]
    latitude:         Optional[float]
    longitude:        Optional[float]
    hotels:           Optional[List[Dict]]
    weather:          Optional[Dict[str, Any]]
    web_description:  Optional[str]
    web_snippets:     Optional[str]
    Bus_info:         Optional[Dict[str, Any]]
    Train_info:       Optional[str]
    emergency:        Optional[str]      
    result:           Optional[str]
    error:            Optional[str]
    route:            Literal["web","final","bus_info","summerize","end"]


# Pydantic model
class BusDetailsDecision(BaseModel):
    bus_path:      Optional[str] = Field(None)
    time_estimate: str           = Field("unknown")
    cost:          str           = Field("unknown")
    description:   str           = Field("No valid Sri Lankan bus route information found.")


# LLM Instances
router_llm    = ChatGroq(model=GROQ_MODEL,  temperature=0,max_tokens=4096)
summarize_llm = ChatGroq(model=GROQ_MODEL,  temperature=0)
bus_llm_raw   = ChatGroq(model=GROQ_MODEL,  temperature=0.4, max_tokens=500)


# Tool map 
TOOLS = [
    GetGeocodingOfLocation,
    GetWeatherOfDestination,
    GetHotelSearchTool,
    GetTrainScheduleTool,
    GetBusRouteTool,
    EmergencyData,
    WebSearchDestinationTool,
]

TOOL_MAP = {t.name: t for t in TOOLS}


# System Prompt
ROUTER_SYSTEM_PROMPT = """
You are a smart Sri Lankan travel assistant. Use tools to answer travel queries.

MEMORY RULES:
- If the user says "same place", "there", "it", "that city" — use the 
  destination from earlier in the conversation.
- If user asks follow-up like "what about hotels?" after asking about 
  weather in Kandy — remember destination is Kandy.
- If user says "and the weather?" after hotels query — reuse last destination.
- Always check conversation history before asking for clarification.

TOOL SELECTION RULES:

RULE 1 — EMERGENCY → EmergencyData
  Keywords: police, tourist police, ambulance, hospital, fire, lost passport,
            theft, embassy, disaster, help, unsafe, emergency, danger
  Call with: user_query = the user's exact message

RULE 2 — WEATHER → GetGeocodingOfLocation THEN GetWeatherOfDestination
  Keywords: weather, rain, raining, temperature, hot, cold, humid,
            sunny, cloudy, forecast, umbrella, monsoon, climate

RULE 3 — HOTELS → GetGeocodingOfLocation THEN GetHotelSearchTool
  Keywords: hotels, accommodation, resorts, guesthouses, where to stay,
            book a room, places to sleep

RULE 4 — TRAINS → GetTrainScheduleTool
  Keywords: train, railway, train schedule, train times, train fare
  Default origin = "Colombo" if not stated

RULE 5 — GENERAL TRAVEL → WebSearchDestinationTool
  Keywords: best places, things to do, travel tips, attractions,
            what to see, travel guide, is it safe

RULE 6 — BUS queries → GetBusRouteTool
  Keywords: bus, by bus, bus fare, bus route, bus schedule,
            bus times, SLTB, CTB, coach, public transport
  Default origin = Colombo if not stated
  Examples:
  - "bus to Kandy"          → GetBusRouteTool(origin="Colombo", destination="Kandy")
  - "bus from Galle"        → GetBusRouteTool(origin="Galle", destination="Colombo")
  - "how to get to Ella"    → GetBusRouteTool + GetTrainScheduleTool (both options)

RULE 7 — HOTELS + WEATHER together → GetGeocodingOfLocation THEN BOTH tools

TOOL CHAINING — CRITICAL:
You will be called multiple times in a loop.
Each time you receive tool results, use them to call the next tool.

Round 1: Call GetGeocodingOfLocation to get coordinates
Round 2: You now have coordinates — call GetHotelSearchTool or GetWeatherOfDestination
Round 3: All data collected — return no tool calls to finish

NEVER call GetHotelSearchTool or GetWeatherOfDestination without real 
coordinates from GetGeocodingOfLocation. Wait for Round 2.

IMPORTANT — TOOL CALL FORMAT:
- Always generate complete valid JSON for tool arguments
- Never truncate or abbreviate argument values
- For GetHotelSearchTool always use exact coordinates from geocoding result
- Keep tool arguments short and precise

COMPLEX QUERIES — handle one tool at a time:
If user asks for weather + hotels + trains + activities:
  Round 1: GetGeocodingOfLocation only
  Round 2: GetWeatherOfDestination + GetHotelSearchTool  
  Round 3: GetTrainScheduleTool
  Round 4: WebSearchDestinationTool
  Never call more than 2 tools per round.
"""


# NODE 1: ROUTER (tool selection + execution)
def router_node(state: AgentState) -> AgentState:
    print("---Enter Router Node---")
    try:
        all_messages = state.get("messages", [])
        
        user_query = next(
            (m.content for m in reversed(all_messages) if isinstance(m, HumanMessage)), ""
        )
        print(f"💬 Query: {user_query}")
        print(f"📚 Context messages: {len(all_messages)}")

        destination = state.get("destination")
        latitude    = state.get("latitude")
        longitude   = state.get("longitude")
        hotels      = state.get("hotels", [])
        weather     = state.get("weather", {})
        train_info  = state.get("Train_info", "")
        web_result  = state.get("web_description", "")
        emergency   = state.get("emergency", "")
        bus_info_data = state.get("Bus_info", None)
        route       = "final"

        # messages = [
        #     {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        #     {"role": "user",   "content": user_query},
        # ]
        llm_messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]

        MAX_HISTORY_MESSAGES = 4
        recent_messages = all_messages[-MAX_HISTORY_MESSAGES:] if len(all_messages) > MAX_HISTORY_MESSAGES else all_messages

        # Inject conversation history before current query 
        # for msg in all_messages:
        #     if isinstance(msg, HumanMessage):
        #         # Skip the last human message — added separately below
        #         if msg.content == user_query:
        #             continue
        #         llm_messages.append({"role": "user", "content": msg.content})
        #     elif isinstance(msg, AIMessage):
        #         llm_messages.append({"role": "assistant", "content": msg.content})
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                if msg.content == user_query:
                    continue
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Truncate AI history messages too
                content = msg.content[:300] if msg.content else ""
            llm_messages.append({"role": "assistant", "content": content})

        # Add current query last 
        llm_messages.append({"role": "user", "content": user_query})

        print(f"📨 LLM context size: {len(llm_messages)} messages")
        

        MAX_ITERATIONS = 7

        for iteration in range(MAX_ITERATIONS):
            print(f"🔄 Tool loop iteration {iteration + 1}")

            response = router_llm.bind_tools(TOOLS).invoke(llm_messages)

            # No tool calls → LLM is done
            if not response.tool_calls:
                print("✅ No more tool calls — loop complete")
                break

            print(f"🤖 Tools: {[c['name'] for c in response.tool_calls]}")

            # Append assistant message to history 
            llm_messages.append(
                {
                    "role": "assistant", 
                    "content": response.content or "",
                    "tool_calls": response.tool_calls
                })

            # Execute each tool call 
            for call in response.tool_calls:
                name = call["name"]
                args = call["args"]
                print(f"🔧 Running: {name}  args={args}")

                tool_fn = TOOL_MAP.get(name)
                if not tool_fn:
                    print(f"❌ Unknown tool: {name}")
                    continue

                try:
                    result = tool_fn.invoke(args)
                    print(f"✅ {name} OK")
                except Exception as e:
                    print(f"❌ {name} failed: {e}")
                    result = {"error": str(e)}

                # Capture result into state
                if name == "GetGeocodingOfLocation" and result.get("latitude"):
                    latitude    = result["latitude"]
                    longitude   = result["longitude"]
                    destination = result.get("destination", destination)
                    route       = "bus_info"

                elif name == "GetWeatherOfDestination":
                    weather = result
                    route   = "bus_info"

                elif name == "GetHotelSearchTool":
                    hotels = result
                    route  = "bus_info"

                elif name == "GetTrainScheduleTool":
                    train_info = result
                    route      = "bus_info"
                    
                elif name == "GetBusRouteTool":
                    bus_info_data = {
                        "bus_path":      f"{result.get('origin', '')} → {result.get('destination', '')}",
                        "time_estimate": result.get("duration", "unknown"),
                        "cost":          str(result.get("fare", "unknown")),
                        "description":   result.get("summary", ""),
                    }
                    # bus_result = result
                    # state["Bus_info"] = {
                    #     "bus_path":      f"{result.get('origin')} → {result.get('destination')}",
                    #     "time_estimate": result.get("duration", "unknown"),
                    #     "cost":          str(result.get("fare", "unknown")),
                    #     "description":   result.get("summary", ""),
                    # }
                    # route = "bus_info"
                    route = "final"  # bus info is usually final, but could be followed by web search if user asked "how to get to X by bus and what to do there"

                elif name == "WebSearchDestinationTool":
                    web_result = result
                    route      = "summerize"

                elif name == "EmergencyData":
                    emergency = result
                    route     = "final"
                    

                # Append tool result back to messages 
                # This is what triggers LLM to call GetHotelSearchTool next
                llm_messages.append({
                    "role":         "tool",
                    "tool_call_id": call.get("id", name),
                    "content":      truncate_tool_result(name, result),
                    "name":         name,
                })
        
        try:
            last_result = str(result)
        except NameError:
            last_result = ""
            
        # ai_summary = (
        #     weather.get("summary", "") if isinstance(weather, dict) and weather.get("summary") else
        #     web_result if web_result else
        #     bus_info_data.get("description", "") if bus_info_data                                        else
        #     train_info if train_info else
        #     last_result
        # )
        web_result_str = (
            web_result.get("result", str(web_result)) if isinstance(web_result, dict)
            else str(web_result) if web_result
            else ""
        )

        ai_summary = (
            weather.get("summary", "")            if isinstance(weather, dict) and weather.get("summary") else
            web_result_str                        if web_result_str                                        else
            bus_info_data.get("description", "")  if isinstance(bus_info_data, dict)                      else
            train_info                            if train_info                                            else
            last_result
        )

        updated_messages = list(all_messages) + [
            AIMessage(content=ai_summary or f"Processed query about {destination or 'Sri Lanka'}")
        ]
        
        return {
            **state,
            "messages":          updated_messages,
            "destination":       destination,
            "destination_place": destination,
            "latitude":          latitude,
            "longitude":         longitude,
            "hotels":            hotels,
            "weather":           weather,
            "Train_info":        train_info,
            "Bus_info":          bus_info_data if bus_info_data else None,
            "web_description":   web_result_str if web_result_str else web_result,
            "emergency":         emergency,
            "result":            "",
            "route":             route,
        }

    except Exception as e:
        print(f"⚠️ Router error: {e}")
        traceback.print_exc()
        return {
            **state,
            "route":  "final",
            "result": "Sorry, I could not process your request.",
        }


# NODE 2: Summarize web search
def summarize_node(state: AgentState) -> AgentState:
    print("--- Entering Summarize Node ---")
    snippets = state.get("web_description", "") or state.get("web_snippets", "")
    user_query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    if not snippets:
        return {**state, "route": "bus_info"}

    try:
        result = summarize_llm.invoke(
            f"""You are a Sri Lankan travel assistant. Summarize the following web search results
to answer the user's question clearly and concisely. Focus only on what the user asked.

User Question: {user_query}
Web Results: {snippets}

Give a clear, factual, helpful summary:"""
        )
        print("✅ Summarize done")
        return {**state, "web_description": result.content, "route": "bus_info"}
    except Exception as e:
        print(f"🔥 Summarize error: {e}")
        return {**state, "route": "bus_info"}



# NODE 3: BUS INFO  (Pinecone RAG)
def bus_info_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("--- Entering Bus Info Node ---")
    user_query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    # Skip if clearly not a bus query
    bus_keywords = ["bus", "route", "transport", "how to get", "travel to", "coach", "ctb"]
    if not any(kw in user_query.lower() for kw in bus_keywords):
        print("⏭️ Skipping bus info — not a bus query")
        return {**state, "Bus_info": None, "route": "final"}

    try:
        # RAG: embed + search Pinecone
        vector  = embedder.encode([user_query]).tolist()[0]
        results = index.query(vector=vector, top_k=5, include_metadata=True)
        context = "\n\n---\n\n".join(
            [m["metadata"]["text"] for m in results["matches"] if "text" in m.get("metadata", {})]
        )

        prompt = f"""
You are a Sri Lankan bus transport assistant. Return ONLY strict JSON, no extra text.

Format:
{{
  "bus_path": "string or null",
  "time_estimate": "string",
  "cost": "string",
  "description": "string"
}}

Rules:
1. If valid route found — fill all fields.
2. If only destination mentioned — assume departure = Colombo.
3. If no valid route — return nulls/unknowns.
4. ONLY JSON, no markdown, no explanation.

Context: {context}
User Question: {user_query}
"""
        raw = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a Sri Lankan bus transport assistant. Return only JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        ).choices[0].message.content.strip()

        print(f"Bus raw: {raw[:200]}")

        # Strip markdown fences if present
        import re
        raw = re.sub(r"```json|```", "", raw).strip()

        bus = BusDetailsDecision.parse_raw(raw)
        return {
            **state,
            "Bus_info": {
                "bus_path":      bus.bus_path,
                "time_estimate": bus.time_estimate,
                "cost":          bus.cost,
                "description":   bus.description,
            },
            "route": "final",
        }

    except Exception as e:
        print(f"🔥 Bus info error: {e}")
        return {
            **state,
            "Bus_info": {"bus_path": None, "time_estimate": "unknown",
                         "cost": "unknown", "description": "Bus info unavailable."},
            "route": "final",
        }


#----------ADDED--------------------------------------------------
# def web_search_node(state: AgentState) -> AgentState:
#     print("--- Entering The Web Search Node ---")

#     query = next(
#         (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
#         ""
#     )

#     print(query)

#     snippets = WebSearchDestinationTool.invoke(query)

#     if snippets.startswith("WEB_ERROR::"):
#         print(f"Web Error: {snippets}. Proceeding to answer with limited info.")
#         return {**state, "web": "", "route": "final"}

#     print(f"Web snippets retrieved: {snippets[:200]}...")
#     print("--- Exiting web_node ---")
#     return {**state, "web_description": snippets, "route": "summerize"}
#----------ADDED--------------------------------------------------




# NODE 5: FINAL (generate summary answer)
def final_node(state: AgentState) -> AgentState:
    print("--- Entering Final Node ---")

    emergency   = state.get("emergency")
    hotels      = state.get("hotels")
    weather     = state.get("weather")
    web_desc    = state.get("web_description", "")
    train_info  = state.get("Train_info", "")
    bus_info    = state.get("Bus_info")
    destination = state.get("destination") or state.get("destination_place", "")
    user_query  = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    # Emergency — return directly
    if emergency:
        return {**state, "result": emergency}

    # Build context for LLM summary
    context_parts = [f"User asked: {user_query}", f"Destination: {destination}"]

    if weather and isinstance(weather, dict):
        context_parts.append(f"Weather: {weather.get('summary', str(weather))}")

    if hotels and isinstance(hotels, list):
        hotel_lines = []
        for h in hotels[:5]:
            if isinstance(h, dict) and "name" in h:
                hotel_lines.append(f"  - {h['name']} ⭐{h.get('stars','?')} — {h.get('price','N/A')}")
        if hotel_lines:
            context_parts.append("Hotels found:\n" + "\n".join(hotel_lines))

    if web_desc:
        context_parts.append(f"Web info: {web_desc[:1000]}")

    if train_info:
        context_parts.append(f"Train info:\n{train_info[:500]}")

    if bus_info and bus_info.get("bus_path"):
        context_parts.append(
            f"Bus info: {bus_info['bus_path']} | {bus_info['time_estimate']} | {bus_info['cost']}\n{bus_info['description']}"
        )

    # If no data collected — graceful response
    if len(context_parts) <= 2:
        return {
            **state,
            "result": "I can help with Sri Lankan travel — hotels, weather, trains, buses, attractions and emergency contacts. What would you like to know?",
        }

    # Generate final answer 
    try:
        result = summarize_llm.invoke(
            f"""You are a friendly Sri Lankan travel assistant.
Using the data below, give the user a helpful, conversational answer.
Be concise. Add useful travel tips where relevant.

{chr(10).join(context_parts)}

Your answer:"""
        )
        return {**state, "result": result.content}
    except Exception as e:
        print(f"🔥 Final node error: {e}")
        return {**state, "result": web_desc or train_info or "Could not generate a summary. Please try again."}


def truncate_tool_result(name: str, result: any) -> str:
    """Keep only essential fields per tool — reduce context size."""
    if name == "GetGeocodingOfLocation":
        if isinstance(result, dict):
            return f"latitude={result.get('latitude')}, longitude={result.get('longitude')}, destination={result.get('destination')}"
        return str(result)[:200]

    elif name == "GetWeatherOfDestination":
        if isinstance(result, dict):
            return f"weather={result.get('weather')}, temperature={result.get('temperature')}°C, is_raining={result.get('is_raining')}"
        return str(result)[:200]

    elif name == "GetHotelSearchTool":
        if isinstance(result, list):
            # Only pass hotel count full data already captured in state
            return f"Found {len(result)} hotels near destination."
        return str(result)[:200]

    elif name == "GetTrainScheduleTool":
        return str(result)[:300]    # train info is usually compact

    elif name == "GetBusRouteTool":
        return str(result)[:300]

    elif name == "WebSearchDestinationTool":
        return str(result)[:400]

    elif name == "EmergencyData":
        return str(result)[:200]

    return str(result)[:300]


# Routing Helpers
def route_after_router(state: AgentState) -> Literal[
    #"web", 
    "summerize", "bus_info", "final"]:
    return state.get("route", "final")

def route_after_summarize(state: AgentState) -> Literal["bus_info", "final"]:
    return state.get("route", "bus_info")

def route_after_bus(state: AgentState) -> Literal["final"]:
    return "final"


# Build Graph
def build_graph():
    """
    Graph flow:
    START → router_node
      → summerize (web results) → bus_info → final → END
      → bus_info → final → END
      → final → END
    """
    g = StateGraph(AgentState)

    #Nodes
    g.add_node("router",    router_node)
    g.add_node("summerize", summarize_node)
    g.add_node("bus_info",  bus_info_node)
    g.add_node("final",     final_node)
    
    #----------ADDED--------------------------------------------------
    #g.add_node("web", web_search_node)
    #----------ADDED--------------------------------------------------

    # Edges 
    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "summerize": "summerize",   # web search → summarize → bus → final
            "bus_info":  "bus_info",    # train/hotel/weather → bus → final
            "web": "bus_info", 
            "final":     "final",       # emergency / no tools → final directly
        }
    )

    g.add_conditional_edges(
        "summerize",
        route_after_summarize,
        {
            "bus_info": "bus_info",
            "final":    "final",
        }
    )

    g.add_edge("bus_info", "final")
    g.add_edge("final",    END)

    return g.compile()
