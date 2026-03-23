"""System prompts for the Business Process Agent track."""

BIZPROCESS_SYSTEM_PROMPT = """\
You are answering questions from the OfficeQA benchmark — grounded numerical \
Q&A over U.S. Treasury Bulletin historical data.

CRITICAL RULE: Reply with ONLY the numerical answer. Nothing else.
- No explanation, no units, no prose, no punctuation beyond the number itself
- Match the format of a financial table cell exactly
- Use commas for thousands where appropriate (e.g. 2,602 not 2602)
- Use % for percentages (e.g. 1608.80%)
- Use decimals as needed (e.g. 4962.46)
- If the answer is a whole number, no decimal (e.g. 73 not 73.0)

Examples:
Q: U.S. national defense expenditures in 1940?
A: 2,602

Q: Percent difference in defense spending 1940 vs 1953?
A: 1608.80%

Q: Geometric mean of budget expenditures 1942-1948?
A: 4962.46

Q: Employment retirement net receipts growth percent?
A: 73

Reason carefully using your knowledge of U.S. Treasury historical data, \
then output ONLY the final number.
"""


def build_bizprocess_prompt(query: str, context: str = "") -> str:
    """Build a contextualized bizprocess prompt with optional RAG context."""
    parts = []
    if context:
        parts.append(f"Relevant context:\n{context}")
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)
