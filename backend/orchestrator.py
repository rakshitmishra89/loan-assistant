# backend/orchestrator.py
import time
from backend.schemas import ChatResponse, DecisionModel, ToolResultsModel, RagMetadataModel, GuardrailsModel, LatencyModel
from backend import memory_store
from backend.adapters import guardrails_adapter

# Import our new Autonomous Agents
from backend.agents import intake_agent, retrieval_agent, tool_agent, decision_agent

async def handle_chat(session_id: str, message: str, metadata: dict = None) -> ChatResponse:
    t0 = time.time()
    agent_trace = []
    
    # 1. SECURITY: Input Guardrails
    verdict_in = guardrails_adapter.moderate_input(message)
    if verdict_in["action"] == "BLOCK":
        return _build_safe_response(session_id, "I cannot process that request due to content policy.", verdict_in, t0)
    safe_message = verdict_in.get("redacted_text", message)

    # 2. MEMORY: Load what we know about the user
    state = memory_store.load(session_id)

    # ---------------------------------------------------------
    # AUTONOMOUS AGENT WORKFLOW STARTS HERE
    # ---------------------------------------------------------
    
    # AGENT 1: INTAKE AGENT (Extracts data)
    extracted_data = intake_agent.process(safe_message, state["entities"])
    
    # Update memory with what the Intake Agent found
    if extracted_data.get("loan_amount"): state["entities"]["loan_amount"] = extracted_data["loan_amount"]
    if extracted_data.get("income_monthly"): state["entities"]["income_monthly"] = extracted_data["income_monthly"]
    if extracted_data.get("tenure_months"): state["entities"]["tenure_months"] = extracted_data["tenure_months"]
    if extracted_data.get("age"): state["entities"]["age"] = extracted_data["age"]
    
    # If the Intake Agent says we are missing data, STOP and ask the user for it
    if extracted_data.get("missing_fields"):
        reply = f"To process your request, I still need your: {', '.join(extracted_data['missing_fields'])}."
        decision = DecisionModel(status="NEED_MORE_INFO")
        return _build_response(session_id, reply, decision, ToolResultsModel(), RagMetadataModel(), verdict_in, state, t0)

    # AGENT 2: RETRIEVAL AGENT (Searches policies)
    t_retrieval_start = time.time()
    rag_data = retrieval_agent.process(safe_message)
    rag_model = RagMetadataModel(used=rag_data["used_rag"], top_k=len(rag_data["chunks"]), chunks=rag_data["chunks"])
    
    # AGENT 3: TOOL AGENT (Does the math)
    tool_results_dict = tool_agent.process(state["entities"])
    tool_model = ToolResultsModel(**tool_results_dict)

    # AGENT 4: DECISION AGENT (Makes the final call)
    chat_history = state.get("summary", "No history yet.")
    reply, decision_dict = decision_agent.process(safe_message, tool_results_dict, rag_data, chat_history)
    decision_model = DecisionModel(**decision_dict)

    # ---------------------------------------------------------
    # AUTONOMOUS AGENT WORKFLOW ENDS HERE
    # ---------------------------------------------------------

    # 3. SECURITY: Output Guardrails
    verdict_out = guardrails_adapter.moderate_output(reply)
    if verdict_out["action"] != "ALLOW":
        reply = verdict_out.get("safe_text", "Output redacted for safety.")

    # 4. MEMORY: Save updated state WITH the messages so it can write the summary!
    memory_store.save(session_id, state, safe_message, reply)

    return _build_response(session_id, reply, decision_model, tool_model, rag_model, verdict_in, state, t0, verdict_out)

def _build_response(session_id, reply, decision, tool_model, rag_model, verdict_in, state, t0, verdict_out=None):
    if not verdict_out: verdict_out = {"action": "ALLOW", "categories": []}
    return ChatResponse(
        session_id=session_id, reply=reply, decision=decision, collected_inputs=state["entities"],
        tool_results=tool_model, rag=rag_model,
        guardrails=GuardrailsModel(input_action=verdict_in["action"], output_action=verdict_out["action"], categories=verdict_in.get("categories", [])),
        latency_ms=LatencyModel(end_to_end=round((time.time() - t0)*1000, 2))
    )

def _build_safe_response(session_id, reply, verdict, t0):
    return ChatResponse(
        session_id=session_id, reply=reply, decision=DecisionModel(), tool_results=ToolResultsModel(), rag=RagMetadataModel(),
        guardrails=GuardrailsModel(input_action=verdict["action"], categories=verdict.get("categories", [])),
        latency_ms=LatencyModel(end_to_end=round((time.time() - t0)*1000, 2))
    )