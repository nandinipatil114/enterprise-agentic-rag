"""Load multi-page documents, chunk recursively, embed locally, and persist to local Qdrant."""
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import BASE_DIR, COLLECTION, EMBEDDING_MODEL, QDRANT_PATH, get_embeddings, get_qdrant

# Each document is a list of pages.
SAMPLE_DOCS: dict[str, list[str]] = {
    "hr_policy.md": [
        "# Acme Corp Employee Handbook - Leave\n\n"
        "Full-time employees receive 24 days of paid annual leave per calendar year, accrued at 2 days per month. "
        "Up to 10 unused annual leave days may be carried over to the next year; carried-over days expire on March 31.\n\n"
        "Sick leave is 12 days per year. A medical certificate is required after 3 consecutive sick days.\n\n"
        "Parental leave: 20 weeks of paid leave for the primary caregiver and 8 weeks for the secondary caregiver.",
        "# Acme Corp Employee Handbook - Work Arrangements\n\n"
        "Acme operates a hybrid model: employees must be in the office at least 3 days per week. "
        "Fully remote arrangements require written approval from a Vice President.\n\n"
        "Home office support: a one-time stipend of $750 and a $50 monthly internet reimbursement.\n\n"
        "New hires serve a 90-day probation period. The notice period is 60 days for confirmed employees "
        "and 30 days during probation.",
    ],
    "it_security_policy.md": [
        "# IT Security Policy - Access Control\n\n"
        "Passwords must be at least 14 characters long and rotated every 90 days. Multi-factor authentication (MFA) "
        "is mandatory for all corporate systems. Hardware security keys are required for production access.\n\n"
        "All laptops must use full-disk encryption. A lost or stolen device must be reported to "
        "security@acme.example within 1 hour of discovery.",
        "# IT Security Policy - Incidents and Data\n\n"
        "Incident response targets: Severity 1 incidents must be acknowledged within 15 minutes, Severity 2 within "
        "1 hour, and Severity 3 within 1 business day. A post-incident review is due within 5 business days.\n\n"
        "Data is classified as Public, Internal, Confidential, or Restricted. Restricted data must be encrypted with "
        "AES-256 at rest and must never be stored on personal devices. Vendor access is reviewed quarterly.",
    ],
    "expense_policy.md": [
        "# Travel and Expense Policy - Limits\n\n"
        "Meals while traveling are reimbursed up to $75 per day. Hotels are capped at $250 per night in tier-1 cities "
        "(New York, San Francisco, London) and $180 per night elsewhere.\n\n"
        "Flights longer than 6 hours may be booked in premium economy. Business class requires CFO approval.",
        "# Travel and Expense Policy - Process\n\n"
        "Expense reports must be submitted within 30 days of the expense and are reimbursed within 10 business days. "
        "Receipts are required for any item over $25.\n\n"
        "Expenses above $500 require manager pre-approval; expenses above $5,000 require Director pre-approval. "
        "Client entertainment is capped at $150 per person.",
    ],
    "cloud_sla.md": [
        "# Acme Cloud Platform SLA - Availability\n\n"
        "Monthly uptime commitments: 99.9% for the Standard plan and 99.99% for the Enterprise plan.\n\n"
        "Service credits: 10% of the monthly fee if uptime is below the commitment but at least 99.0%; "
        "25% if uptime is below 99.0% but at least 95.0%; 50% if uptime is below 95.0%. "
        "Credits must be claimed within 30 days of the end of the affected month.",
        "# Acme Cloud Platform SLA - Support and Recovery\n\n"
        "Support response for Priority 1 issues: 30 minutes, 24/7 for Enterprise; 4 hours during business hours "
        "for Standard.\n\n"
        "Backups run daily and are retained for 35 days. Enterprise recovery objectives: RPO of 1 hour and "
        "RTO of 4 hours.",
    ],
}


def load_documents() -> list[dict]:
    """Sample pages plus any .txt/.md files in ./data (pages separated by form-feed)."""
    pages = [
        {"source": name, "page": i + 1, "text": text}
        for name, doc_pages in SAMPLE_DOCS.items()
        for i, text in enumerate(doc_pages)
    ]
    folder = Path(BASE_DIR / "data")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in {".txt", ".md"}:
                for i, text in enumerate(f.read_text(encoding="utf-8").split("\f")):
                    if text.strip():
                        pages.append({"source": f.name, "page": i + 1, "text": text})
    return pages


def main() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=80, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: list[dict] = []
    for page in load_documents():
        for i, piece in enumerate(splitter.split_text(page["text"])):
            chunks.append({**page, "text": piece, "chunk": i})
    print(f"Prepared {len(chunks)} chunks from {len({c['source'] for c in chunks})} documents")

    embeddings = get_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 64):
        vectors.extend(embeddings.embed_documents([c["text"] for c in chunks[start : start + 64]]))

    client = get_qdrant()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['source']}:{c['page']}:{c['chunk']}")),
                vector=v,
                payload={"text": c["text"], "source": c["source"], "page": c["page"]},
            )
            for c, v in zip(chunks, vectors)
        ],
    )
    total = client.count(COLLECTION).count
    client.close()
    print(f"Indexed {total} vectors (dim={len(vectors[0])}, model={EMBEDDING_MODEL}) into '{COLLECTION}' at {QDRANT_PATH}")


if __name__ == "__main__":
    main()