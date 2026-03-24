# backend/orchestrator.py
import time
from backend.schemas import ChatResponse, DecisionModel, ToolResultsModel, RagMetadataModel, GuardrailsModel, LatencyModel
from backend import memory_store
from backend.adapters import guardrails_adapter

# Import our new Autonomous Agents
from backend.agents import intake_agent, retrieval_agent, tool_agent, decision_agent

async def handle_chat(session_id: str, message: str, metadata: dict = None) -> ChatResponse:
    t0 = time.time()
    agent_trace = [] # <--- THIS IS OUR NEW TRACKER
    
    # 1. SECURITY: Input Guardrails
    verdict_in = guardrails_adapter.moderate_input(message)
    if verdict_in["action"] == "BLOCK":
        return _build_safe_response(session_id, "I cannot process that request due to content policy.", verdict_in, t0)
    safe_message = verdict_in.get("redacted_text", message)
    agent_trace.append({"step": 1, "agent": "🛡️ Guardrails", "action": "Scanned Input", "data": verdict_in})

    # 2. MEMORY: Load state
    state = memory_store.load(session_id)

    # AGENT 1: INTAKE AGENT
    extracted_data = intake_agent.process(safe_message, state["entities"])
    agent_trace.append({"step": 2, "agent": "📥 Intake Agent", "action": "Extracted JSON Entities", "data": extracted_data})
    
    # Update memory
    if extracted_data.get("loan_amount"): state["entities"]["loan_amount"] = extracted_data["loan_amount"]
    if extracted_data.get("income_monthly"): state["entities"]["income_monthly"] = extracted_data["income_monthly"]
    if extracted_data.get("tenure_months"): state["entities"]["tenure_months"] = extracted_data["tenure_months"]
    if extracted_data.get("age"): state["entities"]["age"] = extracted_data["age"]
    if extracted_data.get("credit_score"): state["entities"]["credit_score"] = extracted_data["credit_score"]
    
    if extracted_data.get("missing_fields"):
        reply = f"To process your request, I still need your: {', '.join(extracted_data['missing_fields'])}."
        decision = DecisionModel(status="NEED_MORE_INFO")
        return _build_response(session_id, reply, decision, ToolResultsModel(), RagMetadataModel(), verdict_in, state, t0, agent_trace=agent_trace)

    # AGENT 2: RETRIEVAL AGENT
    rag_data = retrieval_agent.process(safe_message)
    rag_model = RagMetadataModel(used=rag_data["used_rag"], top_k=len(rag_data["chunks"]), chunks=rag_data["chunks"])
    agent_trace.append({"step": 3, "agent": "📚 Retrieval Agent", "action": f"Searched Vector DB. Found {rag_model.top_k} chunks.", "data": [c['source'] for c in rag_data.get('chunks', [])]})

    # AGENT 3: TOOL AGENT
    tool_results_dict = tool_agent.process(state["entities"])
    tool_model = ToolResultsModel(**tool_results_dict)
    agent_trace.append({"step": 4, "agent": "🧮 Tool Agent", "action": "Ran Financial Mathematics", "data": tool_results_dict})

    # AGENT 4: DECISION AGENT
    chat_history = state.get("summary", "No history yet.")
    reply, decision_dict = decision_agent.process(safe_message, tool_results_dict, rag_data, chat_history)
    decision_model = DecisionModel(**decision_dict)
    agent_trace.append({"step": 5, "agent": "⚖️ Decision Agent", "action": "Made Final Credit Verdict", "data": decision_dict})

    # 3. SECURITY: Output Guardrails
    verdict_out = guardrails_adapter.moderate_output(reply)
    if verdict_out["action"] != "ALLOW":
        reply = verdict_out.get("safe_text", "Output redacted for safety.")

    # 4. MEMORY: Save updated state
    memory_store.save(session_id, state, safe_message, reply)

    return _build_response(session_id, reply, decision_model, tool_model, rag_model, verdict_in, state, t0, verdict_out, agent_trace)

# Update the build response to accept the trace
def _build_response(session_id, reply, decision, tool_model, rag_model, verdict_in, state, t0, verdict_out=None, agent_trace=None):
    if not verdict_out: verdict_out = {"action": "ALLOW", "categories": []}
    if not agent_trace: agent_trace = []
    
    return ChatResponse(
        session_id=session_id, reply=reply, decision=decision, collected_inputs=state["entities"],
        tool_results=tool_model, rag=rag_model,
        guardrails=GuardrailsModel(input_action=verdict_in["action"], output_action=verdict_out["action"], categories=verdict_in.get("categories", [])),
        agent_trace=agent_trace, # <--- PASSING IT TO THE FRONTEND
        latency_ms=LatencyModel(end_to_end=round((time.time() - t0)*1000, 2))
    )

def _build_safe_response(session_id, reply, verdict, t0):
    return ChatResponse(
        session_id=session_id, reply=reply, decision=DecisionModel(), tool_results=ToolResultsModel(), rag=RagMetadataModel(),
        guardrails=GuardrailsModel(input_action=verdict["action"], categories=verdict.get("categories", [])),
        latency_ms=LatencyModel(end_to_end=round((time.time() - t0)*1000, 2))
    )