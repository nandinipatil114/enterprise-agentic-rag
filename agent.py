"""LangGraph agentic RAG on Groq: router -> retrieve -> grade -> (rewrite loop) -> generate."""
import ast
import json
import operator
from functools import lru_cache
from typing import Annotated, Iterator, TypedDict

from groq import BadRequestError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import COLLECTION, get_embeddings, get_llm, get_qdrant

MAX_REWRITES = 2
TOP_K = 4

ROUTER_PROMPT = (
    "You are the routing brain of an enterprise assistant for Acme Corp.\n"
    "- For ANY question about company policies, HR, IT security, expenses, travel, or the cloud platform SLA, "
    "call `search_knowledge_base` with a standalone, keyword-rich query (resolve pronouns using the conversation).\n"
    "- For arithmetic, call `calculator`.\n"
    "- For greetings or small talk, answer directly without tools.\n"
    "Never answer policy questions from memory. Call at most one tool at a time."
)
GRADER_PROMPT = (
    "You grade retrieved chunks. Answer relevant=true if the chunk contains facts that help answer the "
    "question, even partially. Otherwise relevant=false."
)
REWRITE_PROMPT = (
    "A search over an enterprise policy knowledge base returned nothing relevant. Rewrite the user's question "
    "as a better search query using precise policy vocabulary and synonyms. Output ONLY the new query."
)
GENERATE_PROMPT = (
    "You are an enterprise policy assistant. Answer using ONLY the numbered context below. Be concise and precise, "
    "cite sources inline like [1], and say so if the context is insufficient.\n\nContext:\n{context}"
)
NOT_FOUND = "I couldn't find relevant information in the knowledge base to answer that reliably."


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    search_query: str
    documents: list[dict]
    rewrite_count: int
    answer: str


class Grade(BaseModel):
    relevant: bool = Field(description="True if the chunk helps answer the question")


# ----------------------------------------------------------------------------- tools
def search_qdrant(query: str, k: int = TOP_K) -> list[dict]:
    vector = get_embeddings().embed_query(query)
    hits = get_qdrant().query_points(
        collection_name=COLLECTION, query=vector, limit=k, with_payload=True
    ).points
    return [
        {"text": h.payload["text"], "source": h.payload["source"], "page": h.payload["page"], "score": round(h.score, 4)}
        for h in hits
    ]


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("Unsupported expression")

    return ev(ast.parse(expression, mode="eval").body)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the enterprise knowledge base (HR, IT security, travel/expense policies, cloud SLA)."""
    return "\n\n".join(d["text"] for d in search_qdrant(query))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '24 * 0.25' or '(750 + 50 * 12)'."""
    return str(_safe_eval(expression))


# ----------------------------------------------------------------------------- nodes
def agent(state: AgentState) -> dict:
    """LLM router: decides between knowledge-base search, calculator, or a direct answer."""
    llm = get_llm().bind_tools([search_knowledge_base, calculator])
    try:
        response = llm.invoke([SystemMessage(ROUTER_PROMPT), *state["messages"]])
    except BadRequestError:
        # Groq occasionally rejects a malformed tool call; fall back to a plain knowledge-base search.
        response = AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": state["question"]},
                         "id": "call_fallback", "type": "tool_call"}],
        )
    if len(response.tool_calls) > 1:  # keep one tool call per step so every call gets a tool result
        response = AIMessage(content=response.content, tool_calls=response.tool_calls[:1])
    update: dict = {"messages": [response]}
    if response.tool_calls:
        call = response.tool_calls[0]
        if call["name"] == "search_knowledge_base":
            update.update(
                search_query=call["args"].get("query") or state["question"], documents=[], rewrite_count=0
            )
    else:
        update["answer"] = str(response.content)
    return update


def route_after_agent(state: AgentState) -> str:
    calls = state["messages"][-1].tool_calls
    if not calls:
        return "end"
    return "retrieve" if calls[0]["name"] == "search_knowledge_base" else "calculate"


def calculate(state: AgentState) -> dict:
    call = state["messages"][-1].tool_calls[0]
    try:
        result = calculator.invoke(call["args"])
    except Exception as exc:  # surface tool errors to the LLM instead of crashing the graph
        result = f"Error: {exc}"
    return {"messages": [ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])]}


def retrieve(state: AgentState) -> dict:
    return {"documents": search_qdrant(state["search_query"])}


def grade_documents(state: AgentState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"documents": []}
    grader = get_llm().with_structured_output(Grade, method="function_calling")
    prompts = [
        [SystemMessage(GRADER_PROMPT), HumanMessage(f"Question: {state['question']}\n\nChunk:\n{d['text']}")]
        for d in docs
    ]
    grades = grader.batch(prompts, config={"max_concurrency": 2}, return_exceptions=True)
    kept = [d for d, g in zip(docs, grades) if isinstance(g, Grade) and g.relevant]
    if not kept and all(not isinstance(g, Grade) for g in grades):
        kept = docs  # grader unavailable (e.g. rate limit): keep chunks rather than fail the answer
    return {"documents": kept}


def route_after_grading(state: AgentState) -> str:
    if state["documents"] or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "generate"
    return "rewrite"


def rewrite_query(state: AgentState) -> dict:
    msg = get_llm().invoke(
        [
            SystemMessage(REWRITE_PROMPT),
            HumanMessage(f"Original question: {state['question']}\nFailed search query: {state['search_query']}"),
        ]
    )
    return {"search_query": str(msg.content).strip().strip('"'), "rewrite_count": state.get("rewrite_count", 0) + 1}


def generate(state: AgentState) -> dict:
    docs = state.get("documents", [])
    pending = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls)
    if docs:
        context = "\n\n".join(f"[{i}] ({d['source']}, p.{d['page']})\n{d['text']}" for i, d in enumerate(docs, 1))
        history = [
            m for m in state["messages"]
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
        ]
        answer = str(get_llm().invoke([SystemMessage(GENERATE_PROMPT.format(context=context)), *history]).content)
    else:
        context, answer = "No relevant context found.", NOT_FOUND
    tool_msg = ToolMessage(content=context, tool_call_id=pending.tool_calls[0]["id"], name=pending.tool_calls[0]["name"])
    return {"messages": [tool_msg, AIMessage(content=answer)], "answer": answer}


# ----------------------------------------------------------------------------- graph
@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(AgentState)
    for name, fn in [
        ("agent", agent), ("calculate", calculate), ("retrieve", retrieve),
        ("grade_documents", grade_documents), ("rewrite_query", rewrite_query), ("generate", generate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"retrieve": "retrieve", "calculate": "calculate", "end": END})
    g.add_edge("calculate", "agent")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading, {"generate": "generate", "rewrite": "rewrite_query"})
    g.add_edge("rewrite_query", "retrieve")  # self-correction loop
    g.add_edge("generate", END)
    return g.compile()


def stream_agent(question: str, history: list[BaseMessage] | None = None) -> Iterator[dict]:
    """Yield per-node state updates in real time: {node_name: update_dict}."""
    inputs = {"messages": [*(history or []), HumanMessage(question)], "question": question}
    yield from get_graph().stream(inputs, stream_mode="updates")


def answer_question(question: str, history: list[BaseMessage] | None = None) -> dict:
    """Run the graph to completion; return answer, graded contexts, rewrite count, and node trace."""
    out = {"answer": "", "contexts": [], "rewrites": 0, "trace": []}
    for step in stream_agent(question, history):
        for node, upd in step.items():
            upd = upd or {}
            out["trace"].append(node)
            out["answer"] = upd.get("answer", out["answer"])
            if node == "rewrite_query":
                out["rewrites"] = upd.get("rewrite_count", out["rewrites"])
            if node == "grade_documents":
                out["contexts"] = [d["text"] for d in upd.get("documents", [])]
    return out


if __name__ == "__main__":
    print(json.dumps(answer_question("How many annual leave days can I carry over?"), indent=2))