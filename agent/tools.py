import uuid

from langchain_core.tools import tool

from agent.fake_db import (
    INVOICES,
    CreditTimeout,
    InvoiceNotFound,
    load_credits,
    save_credit,
)

SAFE_FIELDS = (
    "invoice_id",
    "amount_pence",
    "status",
    "issued_at",
    "plan",
)

TIMEOUT_INVOICES = {"INV-7007"}

_RETRIEVER = None

def _credit_key(invoice_id: str) -> str:
    """One stable key per invoice. Same key = same credit = never issued twice."""
    return f"credit-{invoice_id}"

def _is_credited(invoice: dict) -> bool:
    """Truth comes from two places, read live: the fixture flag, and the credit log."""
    return invoice['already_credited'] or _credit_key(invoice['invoice_id']) in load_credits()


def redact(invoice: dict) -> dict:
    """Keep only the fields a support decision needs.

    Name, email, phone and card never enter the model's context — the
    "no raw rows to the LLM" rule, enforced at the tool boundary rather than
    requested in a prompt.

    Dropping notes also removes the indirect prompt-injection surface: text
    written by a third party cannot influence the model if it is never put in
    front of it. INV-6006's notes field is exactly that attack, and this line
    is the reason it does nothing.
    """
    out = {k: invoice[k] for k in SAFE_FIELDS if k in invoice}
    out['already_credited'] = _is_credited(invoice)
    return out


def _get_retriever():
    """Built once, on first use — importing this module must not build an
index."""
    global _RETRIEVER
    if _RETRIEVER is None:
        from src.retrievers import get_reranked, load_store

        _RETRIEVER = get_reranked(load_store(), k_wide=15, top_n=5)
    return _RETRIEVER


@tool
def search_docs(query: str) -> str:
    """Search the API documentation for how-to and reference information.

    Use this for questions about how to use the API — request bodies, query
    parameters, error codes, authentication, streaming. Do NOT use it for
    anything about a specific invoice or account — use get_invoice for that.
    """
    docs = _get_retriever().invoke(query)
    return "\n\n".join(
        f"[{i}] {d.metadata.get('heading_path',
'')}\n{d.metadata['original_text']}"
        for i, d in enumerate(docs, start=1)
    )


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by its id.

    Returns non-sensitive fields only: amount, status, issue date, plan, and
    whether a credit has already been applied. Customer contact details and
    payment details are never returned and cannot be retrieved through any 
tool.
    """                                                                    
    invoice = INVOICES.get(invoice_id)
    if invoice is None:                                                    
        raise InvoiceNotFound(f"No invoice found with id {invoice_id}")
    return redact(invoice)                                                 

                                                                           
@tool
def issue_credit(invoice_id: str, amount_pence: int) -> dict:              
    """Apply a billing credit against an invoice.
                                                                           
    Idempotent — calling twice for the same invoice credits once and returns
    the original result.                                                   
    """
    if invoice_id in TIMEOUT_INVOICES:
        raise CreditTimeout(f"The billing service timed out for {invoice_id}")

    invoice = INVOICES.get(invoice_id)
    if invoice is None:
        raise InvoiceNotFound(f"No invoice found with id {invoice_id}")

    if invoice["already_credited"]:
        return {"invoice_id": invoice_id, "status": "already_credited", "credit_id": None}

    # Key derived from state, never random — a replay produces the same key,
    # finds the stored record, and issues nothing.
    key = f"credit-{invoice_id}"
    existing = load_credits().get(key)
    if existing:
        return existing

    record = {
        "credit_id": f"CR-{uuid.uuid4().hex[:8]}",
        "invoice_id": invoice_id,
        "amount_pence": amount_pence,
        "status": "issued",
    }
    save_credit(key, record)
    return record