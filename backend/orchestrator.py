# backend/orchestrator.py
import time
import logging
from backend.schemas import ChatResponse, DecisionModel, ToolResultsModel, RagMetadataModel, GuardrailsModel, LatencyModel
from backend import memory_store
from backend.adapters import guardrails_adapter

# Import our new Autonomous Agents
from backend.agents import intake_agent, retrieval_agent, tool_agent, decision_agent

logger = logging.getLogger(__name__)

async def handle_chat(session_id: str, message: str, metadata: dict = None) -> ChatResponse:
    t0 = time.time()
    latency_tracker = {"retrieval": 0.0, "llm": 0.0}  # Track individual component latencies
    agent_trace = []  # Track which agents were involved
    
    # 1. SECURITY: Input Guardrails - ALWAYS runs first
    verdict_in = guardrails_adapter.moderate_input(message)
    if verdict_in["action"] == "BLOCK":
        # Return the safe response from guardrails instead of generic message
        safe_response = guardrails_adapter.get_safe_response(verdict_in.get("categories", ["unknown"])[0] if verdict_in.get("categories") else "unknown")
        return _build_safe_response(session_id, safe_response, verdict_in, t0)
    safe_message = verdict_in.get("redacted_text", message)
    
    # Extract intent hints for routing decisions
    intent_hints = verdict_in.get("intent_hints", {})
    is_security_threat = intent_hints.get("is_security_threat", False)
    is_financial = intent_hints.get("is_financial", False)
    is_policy_query = intent_hints.get("is_policy_query", False)
    is_calculation = intent_hints.get("is_calculation", False)
    intent_confidence = intent_hints.get("confidence", 0.0)
    threat_reason = intent_hints.get("threat_reason")
    
    agent_trace.append({"step": 1, "agent": "Guardrails", "action": "Scanned Input", "data": verdict_in})
    
    # SECURITY CHECK: Block LLM-detected security threats
    if is_security_threat:
        security_response = (
            "Your message has been blocked due to detected security concerns. "
            "This system is designed to help with loan applications and financial queries only. "
            "Any attempts to manipulate, exploit, or compromise this system are logged and monitored. "
            "Please use this service responsibly for legitimate banking inquiries."
        )
        agent_trace.append({
            "step": 2, 
            "agent": "Intent Validator", 
            "action": "Blocked Security Threat", 
            "data": {"reason": threat_reason, "intent_hints": intent_hints}
        })
        return _build_response(
            session_id, security_response, 
            DecisionModel(status="BLOCKED"), ToolResultsModel(), RagMetadataModel(), 
            verdict_in, {"entities": {}}, t0, None, agent_trace
        )


        # OFF-TOPIC CHECK: Block questions unrelated to loans/banking
    is_off_topic = intent_hints.get("is_off_topic", False)
    off_topic_reason = intent_hints.get("off_topic_reason")
    
    if is_off_topic:
        off_topic_response = (
            "I'm a Loan Assistant and can only help with loan-related queries such as:\n"
            "- Loan applications and eligibility\n"
            "- EMI calculations\n"
            "- Interest rates and fees\n"
            "- Policy and documentation questions\n\n"
            "Your question appears to be about something else. "
            "Please ask me about loans or banking services!"
        )
        agent_trace.append({
            "step": 2, 
            "agent": "Intent Validator", 
            "action": "Blocked Off-Topic Query", 
            "data": {"reason": off_topic_reason, "intent_hints": intent_hints}
        })
        return _build_response(
            session_id, off_topic_response, 
            DecisionModel(status="OFF_TOPIC"), ToolResultsModel(), RagMetadataModel(), 
            verdict_in, {"entities": {}}, t0, None, agent_trace
        )
    
    # 2. MEMORY: Load state
    state = memory_store.load(session_id)

    # 2A. INTENT VALIDATION: Check if message is relevant to financial/loan context
    # This prevents off-topic messages from using cached loan data
    if not is_financial and not is_policy_query and not is_calculation and intent_confidence == 0.0:
        # Message has no financial context - check if it's a greeting or completely off-topic
        lower_message = safe_message.lower().strip()
        greeting_patterns = ["hello", "hi", "hey", "help", "what can you do", "thank", "thanks", "bye", "goodbye"]
        is_greeting = any(pattern in lower_message for pattern in greeting_patterns)
        
        if not is_greeting and len(lower_message) > 10:
            # This is an off-topic message that's not a greeting - block it
            off_topic_response = (
                "I'm a Loan Assistant designed to help with:\n"
                "- Loan applications and eligibility checks\n"
                "- EMI calculations\n"
                "- Policy questions about interest rates, fees, and requirements\n\n"
                "Your message doesn't appear to be related to these topics. "
                "How can I assist you with your banking or loan needs?"
            )
            agent_trace.append({
                "step": 2, 
                "agent": "Intent Validator", 
                "action": "Blocked Off-Topic Query", 
                "data": {"reason": "No financial context detected", "intent_hints": intent_hints}
            })
            return _build_response(
                session_id, off_topic_response, 
                DecisionModel(status="OFF_TOPIC"), ToolResultsModel(), RagMetadataModel(), 
                verdict_in, state, t0, None, agent_trace
            )

    # AGENT 1: INTAKE AGENT (with Intent Classification)
    extracted_data = intake_agent.process(safe_message, state["entities"])
    intent = extracted_data.get("intent", "general")
    route_to = extracted_data.get("route_to", "loan_flow")
    agent_trace.append({
        "step": 2, 
        "agent": "Intake Agent", 
        "action": f"Classified Intent: {intent}", 
        "data": {"intent": intent, "route_to": route_to, "extracted": extracted_data}
    })
    
    # Update memory with any extracted data
    if extracted_data.get("loan_amount"): state["entities"]["loan_amount"] = extracted_data["loan_amount"]
    if extracted_data.get("income_monthly"): state["entities"]["income_monthly"] = extracted_data["income_monthly"]
    if extracted_data.get("tenure_months"): state["entities"]["tenure_months"] = extracted_data["tenure_months"]
    if extracted_data.get("age"): state["entities"]["age"] = extracted_data["age"]
    if extracted_data.get("credit_score"): state["entities"]["credit_score"] = extracted_data["credit_score"]
    if extracted_data.get("interest_rate"): state["entities"]["interest_rate"] = extracted_data["interest_rate"]
    
    # ============================================================
    # INTENT-BASED ROUTING
    # ============================================================
    
    # ROUTE 1: POLICY QUESTIONS -> Go directly to RAG
    if route_to == "rag" or intent == "policy_question":
        t_rag_start = time.time()
        rag_data = retrieval_agent.process(safe_message)
        latency_tracker["retrieval"] = (time.time() - t_rag_start) * 1000
        
        rag_model = RagMetadataModel(used=rag_data["used_rag"], top_k=len(rag_data["chunks"]), chunks=rag_data["chunks"])
        agent_trace.append({"step": 3, "agent": "Retrieval Agent", "action": f"RAG Search - Found {rag_model.top_k} chunks", "data": [c['source'] for c in rag_data.get('chunks', [])]})
        
        # Use decision agent to formulate response based on RAG data
        chat_history = state.get("summary", "No history yet.")
        t_llm_start = time.time()
        reply, decision_dict = decision_agent.process(safe_message, {}, rag_data, chat_history)
        latency_tracker["llm"] = (time.time() - t_llm_start) * 1000
        
        decision_model = DecisionModel(**decision_dict)
        agent_trace.append({"step": 4, "agent": "Decision Agent", "action": "Generated RAG Response", "data": decision_dict})
        
        # Output guardrails
        verdict_out = guardrails_adapter.moderate_output(reply)
        if verdict_out["action"] != "ALLOW":
            reply = verdict_out.get("safe_text", "Output redacted for safety.")
        
        memory_store.save(session_id, state, safe_message, reply)
        return _build_response(session_id, reply, decision_model, ToolResultsModel(), rag_model, verdict_in, state, t0, verdict_out, agent_trace, latency_tracker)
    
    # ROUTE 2: GENERAL QUERIES -> Simple response
    if route_to == "general" or intent == "general":
        reply = "Hello! I'm your Loan Assistant. I can help you with:\n- Applying for a loan\n- Answering questions about our loan policies, interest rates, and eligibility\n- Calculating your EMI\n- Checking your eligibility\n\nHow can I assist you today?"
        decision_model = DecisionModel(status="GREETING")
        agent_trace.append({"step": 3, "agent": "General Response", "action": "Greeting/Help", "data": {}})
        
        memory_store.save(session_id, state, safe_message, reply)
        return _build_response(session_id, reply, decision_model, ToolResultsModel(), RagMetadataModel(), verdict_in, state, t0, None, agent_trace)
    
    # ROUTE 3: CALCULATIONS -> Go directly to Tool Agent (skip RAG for math)
    if route_to == "tools" or intent == "calculation":
        # Run Tool Agent directly with extracted values
        tool_results_dict = tool_agent.process(state["entities"])
        tool_model = ToolResultsModel(**tool_results_dict)
        agent_trace.append({"step": 3, "agent": "Tool Agent", "action": "Ran Financial Calculations", "data": tool_results_dict})
        
        # Decision agent will generate structured response for calculations
        chat_history = state.get("summary", "No history yet.")
        t_llm_start = time.time()
        reply, decision_dict = decision_agent.process(safe_message, tool_results_dict, {"used_rag": False, "chunks": []}, chat_history)
        latency_tracker["llm"] = (time.time() - t_llm_start) * 1000
        
        decision_model = DecisionModel(**decision_dict)
        agent_trace.append({"step": 4, "agent": "Decision Agent", "action": "Generated Calculation Response", "data": decision_dict})
        
        # Output guardrails
        verdict_out = guardrails_adapter.moderate_output(reply)
        if verdict_out["action"] != "ALLOW":
            reply = verdict_out.get("safe_text", "Output redacted for safety.")
        
        memory_store.save(session_id, state, safe_message, reply)
        return _build_response(session_id, reply, decision_model, tool_model, RagMetadataModel(), verdict_in, state, t0, verdict_out, agent_trace, latency_tracker)
    
    # ROUTE 4: LOAN APPLICATION -> Collect data, then process
    if extracted_data.get("missing_fields"):
        reply = f"To process your loan application, I still need the following information: {', '.join(extracted_data['missing_fields'])}."
        decision = DecisionModel(status="NEED_MORE_INFO")
        return _build_response(session_id, reply, decision, ToolResultsModel(), RagMetadataModel(), verdict_in, state, t0, agent_trace=agent_trace)

    # ROUTE 5: FULL LOAN PROCESSING (with all data available)
    # AGENT 2: RETRIEVAL AGENT (for additional context)
    t_rag_start = time.time()
    rag_data = retrieval_agent.process(safe_message)
    latency_tracker["retrieval"] = (time.time() - t_rag_start) * 1000
    
    rag_model = RagMetadataModel(used=rag_data["used_rag"], top_k=len(rag_data["chunks"]), chunks=rag_data["chunks"])
    agent_trace.append({"step": 3, "agent": "Retrieval Agent", "action": f"Searched Vector DB. Found {rag_model.top_k} chunks.", "data": [c['source'] for c in rag_data.get('chunks', [])]})

    # AGENT 3: TOOL AGENT
    tool_results_dict = tool_agent.process(state["entities"])
    tool_model = ToolResultsModel(**tool_results_dict)
    agent_trace.append({"step": 4, "agent": "Tool Agent", "action": "Ran Financial Mathematics", "data": tool_results_dict})

    # AGENT 4: DECISION AGENT
    chat_history = state.get("summary", "No history yet.")
    t_llm_start = time.time()
    reply, decision_dict = decision_agent.process(safe_message, tool_results_dict, rag_data, chat_history)
    latency_tracker["llm"] = (time.time() - t_llm_start) * 1000
    
    decision_model = DecisionModel(**decision_dict)
    agent_trace.append({"step": 5, "agent": "Decision Agent", "action": "Made Final Credit Verdict", "data": decision_dict})

    # 3. SECURITY: Output Guardrails
    verdict_out = guardrails_adapter.moderate_output(reply)
    if verdict_out["action"] != "ALLOW":
        reply = verdict_out.get("safe_text", "Output redacted for safety.")

    # 4. MEMORY: Save updated state
    memory_store.save(session_id, state, safe_message, reply)

    return _build_response(session_id, reply, decision_model, tool_model, rag_model, verdict_in, state, t0, verdict_out, agent_trace, latency_tracker)

# Update the build response to accept the trace and latency tracking
def _build_response(session_id, reply, decision, tool_model, rag_model, verdict_in, state, t0, verdict_out=None, agent_trace=None, latency_tracker=None):
    if not verdict_out: verdict_out = {"action": "ALLOW", "categories": []}
    if not agent_trace: agent_trace = []
    if not latency_tracker: latency_tracker = {"retrieval": 0.0, "llm": 0.0}
    
    end_to_end = round((time.time() - t0) * 1000, 2)
    
    return ChatResponse(
        session_id=session_id, reply=reply, decision=decision, collected_inputs=state["entities"],
        tool_results=tool_model, rag=rag_model,
        guardrails=GuardrailsModel(input_action=verdict_in["action"], output_action=verdict_out["action"], categories=verdict_in.get("categories", [])),
        agent_trace=agent_trace,
        latency_ms=LatencyModel(
            retrieval=round(latency_tracker.get("retrieval", 0.0), 2),
            llm=round(latency_tracker.get("llm", 0.0), 2),
            end_to_end=end_to_end
        )
    )

def _build_safe_response(session_id, reply, verdict, t0):
    return ChatResponse(
        session_id=session_id, reply=reply, decision=DecisionModel(), tool_results=ToolResultsModel(), rag=RagMetadataModel(),
        guardrails=GuardrailsModel(input_action=verdict["action"], categories=verdict.get("categories", [])),
        latency_ms=LatencyModel(end_to_end=round((time.time() - t0) * 1000, 2))
    )
