# # prompt_math.py (replace content with this)
# SYSTEM = (
# "You are a math tutor. The following solved examples are provided ONLY as references."
# "They illustrate useful patterns and solution styles, but their numbers, variables, or results"
# "may differ from the NEW QUESTION. Adapt reasoning as needed."


# )

# RULES = """\
# 1) Identify the core concept shared with the reference examples (e.g., derivative rule, probability formula, geometry relation).
# 2) Apply the concept to the NEW QUESTION’s variables and numbers.
# 3) Show reasoning briefly and clearly.
# """

# EXAMPLE_TMPL = """[E{i}] Example
# Q: {q}
# A: {a}
# """

# def build_prompt(tokenizer, new_question, hits, max_input_tokens, ctx_budget_tokens):
#     # Render up to budgeted example tokens
#     parts, used = [], 0
#     for i, (_, _, q, a, url) in enumerate(hits, 1):
#         block = EXAMPLE_TMPL.format(i=i, q=q, a=a).strip()
#         need = len(tokenizer.encode(block, add_special_tokens=False))
#         if used + need > ctx_budget_tokens: break
#         parts.append(block); used += need

#     examples = "\n\n".join(parts)
#     prompt = (
#         SYSTEM + "\n\n"
#         + examples + "\n\n"
#         + "NEW QUESTION:\n" + new_question.strip() + "\n\n"
#         + "INSTRUCTIONS:\n" + RULES + "\n\n"
#         + "Answer:\n"
#         + "Steps: 1) ... 2) ... 3) ...\n"
#         + "FINAL: "
#     )

#     # Trim if over budget by dropping last examples
#     while len(tokenizer.encode(prompt, add_special_tokens=False)) > max_input_tokens and parts:
#         parts.pop()
#         examples = "\n\n".join(parts)
#         prompt = (
#             SYSTEM + "\n\n" + examples + "\n\n"
#             + "NEW QUESTION:\n" + new_question.strip() + "\n\n"
#             + "INSTRUCTIONS:\n" + RULES + "\n\n"
#             + "Answer (show steps):"
#         )
#     return prompt



# prompt_math.py
from typing import List, Tuple

# SYSTEM = (
#   "You are a precise math tutor. Study the solved examples and then solve the NEW QUESTION. "
#   "Show concise steps. Put the final answer after 'FINAL:'."
# )
SYSTEM = (
"You are a math tutor. The following solved examples are provided ONLY as references."
"They illustrate useful patterns and solution styles, but their numbers, variables, or results"
"may differ from the NEW QUESTION. Adapt reasoning as needed."

)
EXAMPLE_TMPL = """Example {i}
Q: {q}
A: {a}
"""

def build_prompt(tokenizer, new_question: str,
                 hits: List[Tuple[str, float, str, str, str]],
                 max_input_tokens: int,
                 ctx_budget_tokens: int) -> str:
    # hits: list of (doc_id, score, Q, A, url)
    exemplars = []
    used = 0
    for i, (_, _, q, a, _) in enumerate(hits, 1):
        block = EXAMPLE_TMPL.format(i=i, q=q, a=a).strip()
        need = len(tokenizer.encode(block, add_special_tokens=False))
        if used + need > ctx_budget_tokens: break
        exemplars.append(block)
        used += need

    context = "\n\n".join(exemplars)
    prompt = (
        SYSTEM + "\n\n" +
        context + "\n\n" +
        "NEW QUESTION:\n" + new_question + "\n\nAnswer:"
    )
    # prompt = (
    #     SYSTEM + "\n\n" +
    #     # context + "\n\n" +
    #     "NEW QUESTION:\n" + new_question + "\n\nAnswer:"
    # )

    # If we accidentally exceed the full input budget, trim exemplars:
    total = len(tokenizer.encode(prompt, add_special_tokens=False))
    if total > max_input_tokens:
        # trim context down to fit
        while exemplars and total > max_input_tokens:
            exemplars.pop()  # drop least recent example
            context = "\n\n".join(exemplars)
            # prompt = SYSTEM + "\n\n" + context + "\n\n" + "NEW QUESTION:\n" + new_question + "\n\nAnswer:"
            ##No context frm rag
            prompt = SYSTEM + "NEW QUESTION:\n" + new_question + "\n\nAnswer:"

            total = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt
