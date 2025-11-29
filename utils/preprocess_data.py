# from datasets import load_dataset
# import os

# data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

# dataset = load_dataset("math-ai/StackMathQA", "stackmathqa100k")
# dataset = dataset["train"]

# n = len(dataset)

# # Compute slice indices
# test_end = int(0.1 * n)
# ft_end = int(0.8 * n)

# ds_test = dataset.select(range(0, test_end))
# ds_ft = dataset.select(range(test_end, ft_end))
# ds_rag = dataset.select(range(ft_end, n))

# # Save splits to disk (Arrow format)
# test_path = os.path.join(data_dir, "test")
# ft_path = os.path.join(data_dir, "ft")
# rag_path = os.path.join(data_dir, "rag")

# ds_test.save_to_disk(test_path)
# ds_ft.save_to_disk(ft_path)
# ds_rag.save_to_disk(rag_path)

# preprocess_with_latex2mathml.py
from datasets import load_dataset, Dataset
import os
import re
import html
from typing import Dict, List, Tuple, Optional
from latex2mathml.converter import convert as latex2mathml_convert

# ----------------- Config -----------------
MIN_WORDS = 5
MAX_WORDS = 400
BATCH_SIZE = 512

# If True: drop examples that contain any LaTeX expression that fails conversion.
# If False: keep example but log a failure flag and do NOT replace that math with MathML.
DROP_ON_INVALID_LATEX = True

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

# ----------------- Load -----------------
dataset = load_dataset("math-ai/StackMathQA", "stackmathqa100k")
dataset = dataset["train"]
n = len(dataset)

test_end = int(0.1 * n)
ft_end = int(0.8 * n)

ds_test = dataset.select(range(0, test_end))
ds_ft = dataset.select(range(test_end, ft_end))
ds_rag = dataset.select(range(ft_end, n))

# ----------------- Regex for math extraction -----------------
# Non-greedy matches, DOTALL to allow multiline math
RE_INLINE_DOLLAR = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', re.DOTALL)
RE_DISPLAY_DOLLAR = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
RE_BRACKET = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)
RE_PAREN = re.compile(r'\\\((.+?)\\\)', re.DOTALL)

# Combined: keep order of appearance by scanning the text
MATH_PATTERNS = [
    ("display_dollar", RE_DISPLAY_DOLLAR),
    ("inline_dollar", RE_INLINE_DOLLAR),
    ("bracket", RE_BRACKET),
    ("paren", RE_PAREN),
]

# ----------------- Helpers -----------------
def extract_math_expressions(text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
    """
    Returns list of tuples (kind, expression, (start, end))
    in their order of appearance.
    """
    matches = []
    for kind, regex in MATH_PATTERNS:
        for m in regex.finditer(text):
            matches.append((m.start(), kind, m.group(1), (m.start(), m.end())))
    # sort by start index to keep appearance order
    matches.sort(key=lambda t: t[0])
    return [(kind, expr, span) for (_, kind, expr, span) in matches]

def try_convert(expr: str) -> Optional[str]:
    """Attempt to convert LaTeX math expression (without delimiters) to MathML. Return None on failure."""
    try:
        mml = latex2mathml_convert(expr)
        return mml
    except Exception:
        return None

def replace_math_with_mathml(text: str, replacements: List[Tuple[Tuple[int,int], str]]) -> str:
    """
    replacements: list of ((start, end), replacement_str)
    Assumes replacements are non-overlapping and sorted by start ascending.
    Returns new text with replacements applied.
    """
    if not replacements:
        return text
    out_parts = []
    last = 0
    for (start, end), rep in replacements:
        out_parts.append(text[last:start])
        out_parts.append(rep)
        last = end
    out_parts.append(text[last:])
    return "".join(out_parts)

# Normalization helpers (from prior script)
_STRIP_TAGS_RE = re.compile(r"</?(?:p|div|span|br|code|pre)[^>]*>", flags=re.I)
_COLLAPSE_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _STRIP_TAGS_RE.sub(" ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = _COLLAPSE_WS_RE.sub(" ", text).strip()
    return text

# Basic fallback LaTeX balance heuristic (keeps previous conservative check)
def is_balanced_latex_simple(text: str) -> bool:
    dollars = text.count('$')
    if dollars % 2 != 0 and dollars > 0:
        return False
    if text.count(r'\[') != text.count(r'\]'):
        return False
    lbrace = text.count('{')
    rbrace = text.count('}')
    if abs(lbrace - rbrace) > max(2, 0.1 * max(1, (lbrace + rbrace) / 2)):
        return False
    return True

# ----------------- Map function -----------------
def clean_batch_with_math(batch: Dict) -> Dict:
    """
    Input: batch dict with keys like 'question', 'answer' (or 'title', etc).
    Output: normalized question/answer, plus:
      - '__keep' boolean flag
      - 'mathml_expressions' list (list of lists per example)
      - 'latex_conversion_failed' boolean flag
      - 'text_with_mathml' text where converted MathML replaced LaTeX (only for successful conversions)
    """
    # determine which fields are present
    questions = batch.get("question") or batch.get("title") or [""] * len(batch.get("id", []))
    answers = batch.get("answer") or batch.get("answers") or [""] * len(batch.get("id", []))

    out_questions = []
    out_answers = []
    keep_flags = []
    mathml_lists = []
    conversion_failed_flags = []
    text_with_mathml_list = []

    for q_raw, a_raw in zip(questions, answers):
        q = normalize_text(q_raw) if isinstance(q_raw, str) else ""
        a = normalize_text(a_raw) if isinstance(a_raw, str) else ""

        # Basic sanity checks
        if (not isinstance(q, str)) or (not isinstance(a, str)) or len(q.strip()) == 0 or len(a.strip()) == 0:
            keep_flags.append(False)
            out_questions.append(q)
            out_answers.append(a)
            mathml_lists.append([])
            conversion_failed_flags.append(True)
            text_with_mathml_list.append(a)
            continue

        # word count filter
        wc = len(a.split())
        if wc < MIN_WORDS or wc > MAX_WORDS:
            keep_flags.append(False)
            out_questions.append(q)
            out_answers.append(a)
            mathml_lists.append([])
            conversion_failed_flags.append(True)
            text_with_mathml_list.append(a)
            continue

        # LaTeX balancing heuristic (fast reject)
        if ('$' in a or r'\[' in a or r'\(' in a or r'\begin' in a) and (not is_balanced_latex_simple(a)):
            # treat as malformed
            keep_flags.append(False)
            out_questions.append(q)
            out_answers.append(a)
            mathml_lists.append([])
            conversion_failed_flags.append(True)
            text_with_mathml_list.append(a)
            continue

        # Extract math expressions in appearance order from the answer (you may also want question)
        matches = extract_math_expressions(a)
        mathmls_for_example = []
        replacements = []  # ((start,end), replacement_str)
        failed = False

        for kind, expr, (start, end) in matches:
            # expr is inner content; try convert
            mml = try_convert(expr)
            if mml is None:
                failed = True
                # if DROP_ON_INVALID_LATEX: break and mark as invalid
                if DROP_ON_INVALID_LATEX:
                    break
                else:
                    # mark but continue, do not replace this one
                    continue
            else:
                mathmls_for_example.append({"kind": kind, "latex": expr, "mathml": mml})
                # choose a replacement placeholder: the MathML string
                replacements.append(((start, end), mml))

        if failed and DROP_ON_INVALID_LATEX:
            keep_flags.append(False)
            conversion_failed_flags.append(True)
            out_questions.append(q)
            out_answers.append(a)
            mathml_lists.append(mathmls_for_example)
            text_with_mathml_list.append(a)
            continue

        # if we reach here, either no matches failed, or we are not dropping on failure.
        # apply replacements (sorted by start)
        replacements.sort(key=lambda x: x[0][0])
        text_with_mathml = replace_math_with_mathml(a, replacements) if replacements else a

        # Passed all filters -> keep
        keep_flags.append(True)
        conversion_failed_flags.append(False)
        out_questions.append(q)
        out_answers.append(a)
        mathml_lists.append(mathmls_for_example)
        text_with_mathml_list.append(text_with_mathml)

    result = {}
    # preserve outgoing keys; prefer 'question'/'answer' naming
    result["question"] = out_questions
    result["answer"] = out_answers
    result["__keep"] = keep_flags
    result["mathml_expressions"] = mathml_lists
    result["latex_conversion_failed"] = conversion_failed_flags
    result["text_with_mathml"] = text_with_mathml_list
    return result

# ----------------- Filter and save -----------------
def filter_and_log(ds: Dataset, name: str) -> Dataset:
    mapped = ds.map(clean_batch_with_math, batched=True, batch_size=BATCH_SIZE, remove_columns=[])
    total = len(mapped)
    keep_count = sum(mapped["__keep"])
    dropped = total - keep_count
    print(f"[{name}] total={total} keep={keep_count} dropped={dropped} ({dropped/total:.2%})")

    # show some rejected examples
    rejected_idx = [i for i, k in enumerate(mapped["__keep"]) if not k][:8]
    for idx in rejected_idx:
        q_snip = mapped["question"][idx][:200] if idx < len(mapped["question"]) else ""
        a_snip = mapped["answer"][idx][:200] if idx < len(mapped["answer"]) else ""
        failed_flag = mapped["latex_conversion_failed"][idx] if idx < len(mapped["latex_conversion_failed"]) else None
        print(f"REJECTED idx={idx} latex_failed={failed_flag} Q={q_snip!r} A={a_snip!r}")

    # filter out
    cleaned = mapped.filter(lambda ex: ex["__keep"])
    # drop helper column but keep mathml_expressions and text_with_mathml
    cleaned = cleaned.remove_columns(["__keep"])
    return cleaned

# Run on splits
ds_test_clean = filter_and_log(ds_test, "test")
ds_ft_clean = filter_and_log(ds_ft, "ft")
ds_rag_clean = filter_and_log(ds_rag, "rag")

# Save to disk
test_path = os.path.join(data_dir, "test_clean_mathml")
ft_path = os.path.join(data_dir, "ft_clean_mathml")
rag_path = os.path.join(data_dir, "rag_clean_mathml")

ds_test_clean.save_to_disk(test_path)
ds_ft_clean.save_to_disk(ft_path)
ds_rag_clean.save_to_disk(rag_path)

print("Saved cleaned splits (with MathML fields) to disk.")
