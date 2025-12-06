BASE_INSTRUCTION = "You are an NL-to-SQL assistant. Use PostgreSQL 9.5. Concise. No comments."
PROMPT_TEMPLATE = """### Database Schema
{schema}

{instruction}

{icl}
Question: {question}
SQL:
"""

class PromptBuilder:
    """Constructs the final prompt for the LLM."""
    
    def build(self, schema: str, question: str, examples: list) -> str:
        """
        Builds the prompt string.
        Args:
            schema: The CREATE TABLE string.
            question: The natural language question.
            examples: List of few-shot examples (dictionaries).
        """
        icl_str = ""
        if examples:
            icl_str = "\n".join([f"### Example\nQuestion: {ex['question']}\nSQL: {ex['query']}" for ex in examples])

        return PROMPT_TEMPLATE.format(
            schema=schema,
            instruction=BASE_INSTRUCTION,
            icl=icl_str,
            question=question
        ).strip()