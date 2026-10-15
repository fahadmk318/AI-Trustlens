import json
import os
import re
import sys
from typing import Any

import ollama


# ------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------

MODEL = "qwen3:8b"

MEDICAL_CATEGORIES = {
    "definition",
    "symptom",
    "disease",
    "cause",
    "risk_factor",
    "diagnosis",
    "test",
    "treatment",
    "medication",
    "dosage",
    "side_effect",
    "contraindication",
    "prevention",
    "prognosis",
    "anatomy",
    "physiology",
    "nutrition",
    "lifestyle",
    "epidemiology",
    "medical_statistics",
    "emergency_advice",
    "other",
}

QUALIFIER_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bcan\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bunlikely\b",
    r"\busually\b",
    r"\bgenerally\b",
    r"\btypically\b",
    r"\boften\b",
    r"\bsometimes\b",
    r"\brarely\b",
    r"\bin some cases\b",
    r"\bnot always\b",
    r"\bmost\b",
    r"\bsome\b",
    r"\balways\b",
    r"\bunless\b",
    r"\bexcept\b",
    r"\bif\b",
    r"\bwhen\b",
    r"\bonly\b",
    r"\bat least\b",
    r"\bup to\b",
    r"\bmore than\b",
    r"\bless than\b",
]

UNSUPPORTED_QUALIFIERS = [
    "always",
    "never",
    "definitely",
    "certainly",
    "primarily",
]

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bwithout\b",
    r"\bdoesn't\b",
    r"\bdon't\b",
    r"\bdidn't\b",
    r"\bisn't\b",
    r"\baren't\b",
    r"\bwasn't\b",
    r"\bweren't\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bcouldn't\b",
    r"\bshouldn't\b",
    r"\bwouldn't\b",
    r"\bwon't\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdid not\b",
    r"\bis not\b",
    r"\bare not\b",
    r"\bwas not\b",
    r"\bwere not\b",
]


# ------------------------------------------------------------
# TEXT & SOURCE SPAN HELPERS
# ------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Normalize whitespace without rewriting wording."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON returned by the model, recovering from enclosing text."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("The model did not return a valid JSON object.")


def _find_patterns(text: str, patterns: list[str]) -> list[str]:
    """Return distinct patterns present in text, case-insensitively."""
    found = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            found.append(match.group(0).lower())
    return list(dict.fromkeys(found))


def _extract_numbers_and_units(text: str) -> list[str]:
    """
    Extract number-like tokens, including decimals, ranges, temperatures,
    and associated units (e.g., '100.4°f', '36.1°c', '97-99°f').
    """
    pattern = (
        r"(?<!\w)"
        r"\d+(?:,\d{3})*(?:\.\d+)?"
        r"(?:\s*(?:-|–|—|to)\s*\d+(?:,\d{3})*(?:\.\d+)?)?"
        r"\s*(?:°\s*[CF]|mg|mcg|µg|g|kg|ml|mL|L|"
        r"mmHg|bpm|%|years?|months?|weeks?|days?|hours?|minutes?)?"
    )
    tokens = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        tok = re.sub(r"\s+", "", match.group(0)).lower()
        if tok and not tok.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")) is False:
            tokens.append(tok)
    return tokens


def extract_source_spans(doc: str) -> list[dict[str, Any]]:
    """
    Segment the source document into identifiable sentences, bullet points,
    and headers with exact character offsets [start, end].
    Associates list items with their parent heading context.
    Detects whether items contain unfinished formatting or are incomplete fragments.
    """
    if not doc or not doc.strip():
        return []

    split_indices = {0, len(doc)}

    # 1. Headings:
    # a) Markdown hashes: ### Title or # Title
    for m in re.finditer(r"(?:^|[\r\n]|(?<=[.!?:\s]))#{1,6}\s*[^#\n\r\-:]*?:?", doc):
        split_indices.add(m.start())
        split_indices.add(m.end())

    # b) Text headings ending with colon on their own line or before a list
    for m in re.finditer(r"(?:^|[\r\n])\s*([A-Za-z0-9\s\-_]{3,60}:)(?=\s*[\r\n]|\s*[-*+])", doc):
        split_indices.add(m.start())
        split_indices.add(m.end())

    # 2. Bullets:
    # Hyphen bullets (including inline - **item**)
    for m in re.finditer(r"(?:^|(?<=[.!?:\s)*]))-+\s*(?:\*\*)?", doc):
        split_indices.add(m.start())
    # Asterisk / plus / numbered bullets (must be followed by whitespace: * item, + item, 1. item)
    for m in re.finditer(r"(?:^|[\r\n]|(?<=[.!?:\s]))\s*(?:\*|\+|\d+\.)\s+", doc):
        split_indices.add(m.start())

    # 3. Sentences:
    for m in re.finditer(r"(?<!\d)(?:[.!?])(?!\d)(?=\s+[A-Z]|\s*[-*#]|\s*$)", doc):
        split_indices.add(m.end())

    sorted_idx = sorted(split_indices)
    spans: list[dict[str, Any]] = []
    current_header = ""

    for i in range(len(sorted_idx) - 1):
        s_idx = sorted_idx[i]
        e_idx = sorted_idx[i + 1]
        chunk = doc[s_idx:e_idx].strip()
        if not chunk:
            continue

        clean = re.sub(r"^(?:#{1,6}|[-*+]|\d+\.)\s*", "", chunk).strip()
        clean_no_md = re.sub(r"\*\*|__", "", clean).strip()

        if not clean or clean == ":":
            continue

        is_header = chunk.startswith("#") or (chunk.endswith(":") and not chunk.startswith(("-", "*", "+")))
        if is_header:
            current_header = re.sub(r"^#{1,6}\s*", "", clean).rstrip(":").strip()

        is_final = e_idx >= len(doc.rstrip())

        # Check for unclosed markdown / brackets
        has_unclosed_md = (
            clean.count("**") % 2 != 0
            or clean.count("(") != clean.count(")")
            or clean.count("[") != clean.count("]")
        )

        # Check if span has finite verb / predicate
        verb_pattern = r"\b(is|are|was|were|can|could|may|might|must|should|has|have|had|causes?|caused|develops?|leads?|indicates?|occurs?|remains?|considered)\b"
        has_verb = bool(re.search(verb_pattern, clean_no_md, re.IGNORECASE))

        is_truncated = False
        if has_unclosed_md:
            is_truncated = True
        elif is_final and not is_header:
            # Trailing fragment check: incomplete if lacks verb and doesn't end in punctuation
            if not has_verb and not clean.endswith((".", "!", "?", '"', "”")):
                is_truncated = True

        spans.append({
            "start": s_idx,
            "end": e_idx,
            "text": clean,
            "plain": clean_no_md.lower(),
            "is_header": is_header,
            "heading_context": current_header if not is_header else "",
            "enclosing_header": current_header if not is_header else "",
            "is_truncated": is_truncated,
        })

    return spans


def align_claim_to_source(claim_text: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Find the best matching source span for a claim based on bidirectional token containment.
    Returns source_sentence, character offsets, heading_context, and whether the source is truncated.
    """
    claim_words = set(w for w in re.findall(r"\w+", claim_text.lower()) if len(w) > 2)
    if not claim_words or not spans:
        return {
            "source_sentence": "",
            "source_start": None,
            "source_end": None,
            "heading_context": "",
            "enclosing_header": "",
            "is_truncated": False,
            "aligned": False,
        }

    best_span = None
    best_score = 0.0

    for s in spans:
        if s["is_header"]:
            continue
        span_words = set(re.findall(r"\w+", s["plain"]))
        if not span_words:
            continue
        overlap = len(claim_words & span_words)
        claim_coverage = overlap / len(claim_words)
        span_coverage = overlap / len(span_words)
        score = max(claim_coverage, span_coverage)

        if score > best_score:
            best_score = score
            best_span = s

    if best_span and best_score >= 0.35:
        hdr = best_span.get("heading_context") or best_span.get("enclosing_header", "")
        return {
            "source_sentence": best_span["text"],
            "source_start": best_span["start"],
            "source_end": best_span["end"],
            "heading_context": hdr,
            "enclosing_header": hdr,
            "is_truncated": best_span["is_truncated"],
            "aligned": True,
        }

    return {
        "source_sentence": "",
        "source_start": None,
        "source_end": None,
        "heading_context": "",
        "enclosing_header": "",
        "is_truncated": False,
        "aligned": False,
    }


def _safe_claim_item(item: Any) -> dict[str, Any] | None:
    """Validate and normalize one claim object."""
    if not isinstance(item, dict):
        return None

    claim = item.get("claim")
    category = item.get("category", "other")

    if not isinstance(claim, str) or not claim.strip():
        return None

    if not isinstance(category, str):
        category = "other"

    category = category.strip().lower()
    if category not in MEDICAL_CATEGORIES:
        category = "other"

    needs_review = item.get("needs_review", False)
    if not isinstance(needs_review, bool):
        needs_review = True

    reason = item.get("reason", "")
    if not isinstance(reason, str):
        reason = "Model returned an invalid review reason."

    source_sentence = item.get("source_sentence", "")
    if not isinstance(source_sentence, str):
        source_sentence = ""

    heading_context = item.get("heading_context") or item.get("enclosing_header") or ""
    if not isinstance(heading_context, str):
        heading_context = ""

    source_start = item.get("source_start")
    if not isinstance(source_start, int):
        source_start = None

    source_end = item.get("source_end")
    if not isinstance(source_end, int):
        source_end = None

    return {
        "claim": _normalise_text(claim),
        "category": category,
        "source_sentence": _normalise_text(source_sentence),
        "heading_context": _normalise_text(heading_context),
        "source_start": source_start,
        "source_end": source_end,
        "needs_review": needs_review,
        "reason": reason.strip(),
    }


# ------------------------------------------------------------
# STAGE 1: EXTRACT MEDICAL CLAIMS
# ------------------------------------------------------------

def extract_medical_claims(original_answer: str) -> dict[str, Any]:
    """
    Extract atomic claims from a medical answer using local Qwen3.
    Preserves exact certainty, avoids ungrammatical splits, and links source sentences and heading context.
    """
    if not isinstance(original_answer, str) or not original_answer.strip():
        return {"claims": [], "error": "The input answer is empty."}

    source_spans = extract_source_spans(original_answer)

    system_prompt = f"""
You are a precision medical factual-claim extraction system for AI TrustLens.

Your task is to decompose the supplied medical answer into atomic, verifiable factual claims with exact source sentence alignment and parent heading context.

OUTPUT SCHEMA (JSON ONLY):
{{
  "claims": [
    {{
      "claim": "Stand-alone, grammatically complete atomic factual claim",
      "category": "one category from the allowed list",
      "source_sentence": "Exact sentence or bullet point from the source answer",
      "heading_context": "Parent heading if applicable, else empty string",
      "needs_review": false,
      "reason": ""
    }}
  ]
}}

ALLOWED CATEGORIES:
{", ".join(sorted(MEDICAL_CATEGORIES))}

CATEGORY ASSIGNMENT:
- Body temperature thresholds (e.g. 97°F–99°F, 100.4°F) are "physiology", not "anatomy".
- Causes and etiologies are "cause".
- Manifestations and clinical signs are "symptom".
- Outcomes and course of illness are "prognosis".
- If category is genuinely ambiguous, use "other" and set needs_review=true.

CRITICAL EXTRACTION RULES:
1. FAITHFUL FIDELITY (NO SEMANTIC DRIFT):
   - Preserve the exact meaning, certainty, frequency, and strength of each statement.
   - Do NOT add qualifiers like "common", "usually", "always" unless explicitly stated in the source text or justified by its parent heading.
   - For list items under a heading like "Common Causes of Fever:", items may inherit the heading context (e.g. "Infections are a common cause of fever" or "Infections can cause fever"), but do NOT claim "common cause" for items that have restrictive qualifiers (e.g. "Cancer (in some cases)").
   - Preserve all qualifiers: "usually", "typically", "often", "may", "can", "in some cases", "not always", "rarely".
   - Preserve all negations: "not", "no", "never", "without", "neither", "nor".
   - Preserve exact numbers, units, temperature ranges, and thresholds (e.g. "97°F (36.1°C)", "100.4°F (38°C)").

2. ATOMIC DECOMPOSITION & COMPLETE STATEMENTS:
   - Split sentences containing independent assertions joined by contrastive conjunctions ("but", "although", "whereas", "while") or semicolons into distinct claims.
   - EXAMPLE:
     Source: "Fevers are usually not harmful in themselves, but they can be a sign that something is wrong in the body."
     Split into:
     - "Fevers are usually not harmful in themselves." (category: prognosis)
     - "Fevers can be a sign that something is wrong in the body." (category: symptom)
   - DO NOT create ungrammatical fragments!
     Source: "A fever is a temporary increase in body temperature, usually as a response to an illness or infection."
     Keep it as a complete grammatical claim:
     "A fever is a temporary increase in body temperature, usually as a response to an illness or infection."
   - Keep lists of examples together when they illustrate a single claim (e.g. "such as a virus, bacteria, or other foreign substances").

3. INCOMPLETE OR TRUNCATED SOURCE INPUT:
   - If the source text contains an unfinished fragment or unfinished formatting (e.g. "- **Heat exhaustion"), do NOT invent a predicate ("Heat exhaustion is a cause of fever").
   - Preserve the original incomplete fragment as the claim, set needs_review=true, and set reason="Incomplete source statement: fragment lacks predicate or context."

4. SOURCE ALIGNMENT:
   - Copy the exact source sentence or bullet item into "source_sentence".
   - Copy the enclosing section heading (if any) into "heading_context".

5. Return ONLY valid JSON matching the schema. No markdown fences, no conversational text.
"""

    user_prompt = f"MEDICAL ANSWER:\n{original_answer}"

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            think=False,
            options={"temperature": 0, "num_predict": 1200},
        )

        parsed = _extract_json(response["message"]["content"])
        raw_claims = parsed.get("claims", [])

        if not isinstance(raw_claims, list):
            raise ValueError("'claims' must be a JSON list.")

        claims = []
        for raw_item in raw_claims:
            item = _safe_claim_item(raw_item)
            if item is None:
                continue

            # Compute or verify source alignment against document spans
            alignment = align_claim_to_source(item["claim"], source_spans)
            if alignment["aligned"]:
                item["source_sentence"] = alignment["source_sentence"]
                item["source_start"] = alignment["source_start"]
                item["source_end"] = alignment["source_end"]
                if not item.get("heading_context") and alignment.get("heading_context"):
                    item["heading_context"] = alignment["heading_context"]

                if alignment["is_truncated"]:
                    clean_frag = re.sub(r"^\*+|\*+$", "", alignment["source_sentence"]).strip()
                    clean_frag = re.sub(r"^(?:[-*+]|\d+\.)\s*", "", clean_frag).strip()
                    if clean_frag:
                        item["claim"] = clean_frag
                    item["needs_review"] = True
                    item["reason"] = "Incomplete source statement: fragment lacks predicate or context."
            else:
                item["needs_review"] = True
                item["reason"] = "Claim has no reliable source alignment."

            claims.append(item)

        if not claims:
            return {
                "claims": [],
                "error": "The model returned no usable claims.",
            }

        return {"claims": claims, "error": None}

    except Exception as exc:
        return {
            "claims": [],
            "error": (
                f"Claim extraction failed: {exc}. "
                "Check that Ollama is running and the model is installed."
            ),
        }


# ------------------------------------------------------------
# STAGE 2: DECOMPOSE COMPOUND CLAIMS
# ------------------------------------------------------------

def split_compound_claims(
    extraction_result: dict[str, Any],
    source_text: str = "",
) -> dict[str, Any]:
    """
    Decompose any compound claims that were not separated into atomic claims.
    Handles contrastive conjunctions ('but', 'although', 'whereas', 'yet', semicolons)
    while avoiding ungrammatical fragment splitting on adverbial clauses or examples.
    """
    if extraction_result.get("error"):
        return extraction_result

    claims = extraction_result.get("claims", [])
    if not isinstance(claims, list):
        return extraction_result

    source_spans = extract_source_spans(source_text) if source_text else []
    decomposed: list[dict[str, Any]] = []

    for item in claims:
        if not isinstance(item, dict):
            continue

        claim_text = item.get("claim", "").strip()
        category = item.get("category", "other")
        needs_review = item.get("needs_review", False)
        reason = item.get("reason", "")
        source_sentence = item.get("source_sentence", "")
        heading_context = item.get("heading_context", "")
        source_start = item.get("source_start")
        source_end = item.get("source_end")

        # Do NOT split sentences like:
        # "A fever is a temporary increase in body temperature, usually as a response to..."
        # (comma followed by an adverbial modifier without an independent subject/predicate)
        if re.search(r",\s*(?:usually|typically|often|rarely|mostly)\s+(?:as|in|during|after)\b", claim_text, re.IGNORECASE):
            decomposed.append(item)
            continue

        # Do NOT split example lists like "such as a virus, bacteria..."
        if re.search(r",\s*(?:such as|including|for example|e\.g\.)\b", claim_text, re.IGNORECASE):
            decomposed.append(item)
            continue

        # Match contrastive split patterns like ", but ", ", although ", ", whereas ", "; "
        split_match = re.search(
            r"^(.*?)(?:,\s*(?:but|whereas|although|yet|while)|;\s*|,\s*however,\s*)\s*(.*)$",
            claim_text,
            flags=re.IGNORECASE,
        )
        if not split_match:
            split_match = re.search(
                r"^(.*?)\s+(?:but|whereas|although|yet)\s+(.*)$",
                claim_text,
                flags=re.IGNORECASE,
            )

        if split_match:
            part1 = split_match.group(1).strip()
            part2 = split_match.group(2).strip()

            if not part1.endswith((".", "!", "?")):
                part1 += "."
            if not part2.endswith((".", "!", "?")):
                part2 += "."

            # Extract subject from part1
            verb_m = re.search(r"\b(are|is|were|was|can|could|may|might|will|would|have|has|do|does)\b", part1, re.IGNORECASE)
            first_subject = ""
            if verb_m:
                first_subject = part1[:verb_m.start()].strip()
            if not first_subject:
                words1 = part1.split()
                first_subject = words1[0] if words1 else ""

            # Pronoun / elliptical subject resolution: e.g. "Fevers are ... but they can be..."
            if re.match(r"^they\b", part2, flags=re.IGNORECASE) and first_subject:
                part2 = re.sub(r"^they\b", first_subject, part2, flags=re.IGNORECASE)
            elif re.match(r"^it\b", part2, flags=re.IGNORECASE) and first_subject:
                part2 = re.sub(r"^it\b", first_subject, part2, flags=re.IGNORECASE)
            elif re.match(r"^(?:can|could|may|might|will|would|is|are|was|were)\b", part2, flags=re.IGNORECASE) and first_subject:
                part2 = first_subject + " " + part2

            if part2:
                part2 = part2[0].upper() + part2[1:]

            claim_a = {
                "claim": part1,
                "category": category,
                "source_sentence": source_sentence,
                "heading_context": heading_context,
                "source_start": source_start,
                "source_end": source_end,
                "needs_review": needs_review,
                "reason": reason,
            }

            category_b = "symptom" if ("sign" in part2.lower() or "symptom" in part2.lower()) else category
            claim_b = {
                "claim": part2,
                "category": category_b,
                "source_sentence": source_sentence,
                "heading_context": heading_context,
                "source_start": source_start,
                "source_end": source_end,
                "needs_review": needs_review,
                "reason": reason,
            }

            decomposed.append(claim_a)
            decomposed.append(claim_b)
        else:
            decomposed.append(item)

    # Re-verify source alignment if source spans are available
    if source_spans:
        for itm in decomposed:
            if not itm.get("source_sentence"):
                align = align_claim_to_source(itm["claim"], source_spans)
                if align["aligned"]:
                    itm["source_sentence"] = align["source_sentence"]
                    itm["source_start"] = align["source_start"]
                    itm["source_end"] = align["source_end"]
                    if not itm.get("heading_context") and align.get("heading_context"):
                        itm["heading_context"] = align["heading_context"]

    return {"claims": decomposed, "error": None}


# ------------------------------------------------------------
# STAGE 3: REFINE CLAIMS VIA MODEL (OPTIONAL)
# ------------------------------------------------------------

def refine_medical_claims(
    original_answer: str,
    extraction_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Ask the model to review the complete answer and extracted claims
    for compound assertions, omissions, and meaning changes.
    """
    if extraction_result.get("error"):
        return extraction_result

    existing_claims = extraction_result.get("claims", [])
    if not existing_claims:
        return {
            "claims": [],
            "error": "There are no claims to refine.",
        }

    system_prompt = f"""
You refine medical claim decomposition. You are NOT a medical fact checker.

Return JSON only:
{{
  "claims": [
    {{
      "claim": "One independently verifiable assertion",
      "category": "symptom",
      "source_sentence": "Exact source sentence",
      "needs_review": false,
      "reason": ""
    }}
  ]
}}

CATEGORIES:
{", ".join(sorted(MEDICAL_CATEGORIES))}

RULES:
1. Split every sentence containing multiple independently verifiable
   assertions into separate claims.
2. Preserve negation, uncertainty, probability, frequency, conditions,
   scope, comparisons, exceptions, and qualifiers in the appropriate claim.
3. Preserve all numbers, units, thresholds, and ranges.
4. Do not strengthen, weaken, or otherwise change the source meaning.
5. Do not invent facts or use outside medical knowledge.
6. Include all material claims from the original answer and do not duplicate assertions.
7. Output JSON only, without Markdown fences or commentary.
"""

    user_prompt = (
        "ORIGINAL ANSWER:\n"
        + original_answer
        + "\n\nCURRENT CLAIMS:\n"
        + json.dumps(existing_claims, ensure_ascii=False)
        + "\n\nReturn the complete refined claim list."
    )

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            think=False,
            options={"temperature": 0, "num_predict": 1200},
        )

        parsed = _extract_json(response["message"]["content"])
        raw_claims = parsed.get("claims", [])

        if not isinstance(raw_claims, list):
            raise ValueError("'claims' must be a JSON list.")

        claims = []
        for raw_item in raw_claims:
            item = _safe_claim_item(raw_item)
            if item is not None:
                claims.append(item)

        if not claims:
            raise ValueError("The refinement returned no usable claims.")

        return {"claims": claims, "error": None}

    except Exception as exc:
        return {
            "claims": existing_claims,
            "error": f"Refinement failed; original claims retained: {exc}",
        }


# ------------------------------------------------------------
# STAGE 4: STRUCTURAL / FIDELITY VALIDATION
# ------------------------------------------------------------

def validate_extracted_claims(
    original_answer: str,
    extraction_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic structural and source-alignment fidelity validation.
    Compares each claim specifically against its own aligned source statement.
    Categorizes status as PASSED, REVIEW_REQUIRED, or FAILED.
    """
    if not isinstance(original_answer, str) or not original_answer.strip():
        return {
            "valid": False,
            "claims": [],
            "issues": ["Original answer is empty."],
            "summary": {"passed": 0, "review_required": 0, "failed": 1},
        }

    raw_claims = extraction_result.get("claims", [])
    all_issues = []
    checked_claims = []

    if extraction_result.get("error"):
        all_issues.append(f"Pipeline error: {extraction_result['error']}")

    if not isinstance(raw_claims, list) or not raw_claims:
        return {
            "valid": False,
            "claims": [],
            "issues": all_issues + ["No claims were supplied for validation."],
            "summary": {"passed": 0, "review_required": 0, "failed": 1},
        }

    source_spans = extract_source_spans(original_answer)
    seen_normalized_claims: set[str] = set()

    for index, raw_item in enumerate(raw_claims, start=1):
        item = _safe_claim_item(raw_item)
        issues: list[str] = []
        is_hard_failure = False

        if item is None:
            msg = f"Claim {index}: Malformed claim object or empty text."
            all_issues.append(msg)
            checked_claims.append({
                "claim": str(raw_item),
                "category": "other",
                "source_sentence": "",
                "source_start": None,
                "source_end": None,
                "validation": "FAILED",
                "issues": [msg],
            })
            continue

        claim = item["claim"]
        claim_lower = claim.lower()
        norm_key = re.sub(r"[^\w\s]", "", claim_lower).strip()

        # 1. Duplicate Claim Check
        if norm_key in seen_normalized_claims:
            issues.append("Duplicate claim: identical assertion already extracted.")
        seen_normalized_claims.add(norm_key)

        # 2. Source Alignment Verification
        source_sentence = item["source_sentence"]
        if not source_sentence:
            alignment = align_claim_to_source(claim, source_spans)
            if alignment["aligned"]:
                source_sentence = alignment["source_sentence"]
                item["source_sentence"] = source_sentence
                item["source_start"] = alignment["source_start"]
                item["source_end"] = alignment["source_end"]
                if alignment["is_truncated"]:
                    issues.append("Source statement is an incomplete fragment lacking predicate or context.")
            else:
                issues.append("Unsupported claim text: cannot establish reliable alignment to source sentence.")
        matched_span = next(
            (sp for sp in source_spans if sp["text"] == source_sentence or sp["start"] == item.get("source_start")),
            None,
        )
        if matched_span and matched_span.get("is_truncated"):
            issues.append("Source statement is an incomplete fragment lacking predicate or context.")
        elif item.get("needs_review") and "incomplete" in item.get("reason", "").lower():
            issues.append("Source statement is an incomplete fragment lacking predicate or context.")

        if matched_span and (matched_span["text"].count("**") % 2 != 0 or matched_span["text"].count("(") != matched_span["text"].count(")")):
            issues.append("Unfinished Markdown formatting in source statement.")

        heading_context = item.get("heading_context") or (matched_span.get("heading_context", "") if matched_span else "")
        header_lower = heading_context.lower() if heading_context else ""

        # For compound sentences with contrastive conjunctions, check against the matching clause
        check_context = source_sentence
        if source_sentence and re.search(r",\s*(?:but|whereas|although|yet)\s*", source_sentence, re.IGNORECASE):
            clauses = re.split(r",\s*(?:but|whereas|although|yet)\s*", source_sentence, flags=re.IGNORECASE)
            c_words = set(re.findall(r"\w+", claim_lower))
            best_cl = source_sentence
            best_ov = 0
            for cl in clauses:
                cl_w = set(re.findall(r"\w+", cl.lower()))
                ov = len(c_words & cl_w)
                if ov > best_ov:
                    best_ov = ov
                    best_cl = cl
            check_context = best_cl

        src_lower = source_sentence.lower() if source_sentence else ""
        ctx_lower = check_context.lower() if check_context else ""
        combined_lower = f"{header_lower} {src_lower}".strip()

        # 3. Incomplete Grammatical Fragment Check
        fragment_patterns = [
            r"^is\s+(?:usually|typically|often|a|an)\b",
            r"^as\s+a\s+response\b",
            r"^usually\s+as\s+a\s+response\b",
        ]
        for fp in fragment_patterns:
            if re.search(fp, claim_lower):
                issues.append("Incomplete grammatical fragment: claim lacks an independent subject/predicate.")
                is_hard_failure = True
                break

        # 4. Potentially Compound Claim Check
        compound_patterns = [
            (r"\bbut\b", "Contains 'but'; assertion should be decomposed into independent atomic claims."),
            (r"\balthough\b", "Contains 'although'; assertion should be decomposed."),
            (r"\bwhereas\b", "Contains 'whereas'; assertion should be decomposed."),
            (r";", "Contains semicolon; assertion should be decomposed."),
        ]
        for pattern, msg in compound_patterns:
            if re.search(pattern, claim_lower):
                issues.append(f"Potentially compound claim: {msg}")
                break

        # 5. Number and Unit Fidelity Check (against its OWN source_sentence)
        if source_sentence:
            claim_numbers = _extract_numbers_and_units(claim)
            source_numbers = _extract_numbers_and_units(source_sentence)
            for num in claim_numbers:
                if num not in source_numbers:
                    issues.append(f"Altered or ungrounded numeric token '{num}': not found in aligned source sentence.")
                    is_hard_failure = True

        # 6. Negation Fidelity Check (against its matched clause)
        if check_context:
            ctx_negations = _find_patterns(check_context, NEGATION_PATTERNS)
            claim_negations = _find_patterns(claim, NEGATION_PATTERNS)
            if ctx_negations and not claim_negations:
                words_overlap = [w for w in re.findall(r"\w+", claim_lower) if len(w) > 3 and w in ctx_lower]
                if len(words_overlap) >= 3:
                    issues.append("Negation mismatch: source statement contains negation not preserved in claim.")
                    is_hard_failure = True

        # 7. Semantic Drift & Qualifier Fidelity (Modality, Frequency, Certainty)
        if check_context or heading_context:
            for sq in ["in some cases", "not always", "rarely", "usually", "typically"]:
                if re.search(r"\b" + re.escape(sq) + r"\b", ctx_lower):
                    if not re.search(r"\b" + re.escape(sq) + r"\b", claim_lower):
                        issues.append(f"Scope qualifier dropped: '{sq}' present in source statement was omitted.")

            # Check "common"
            if re.search(r"\bcommon\b", claim_lower):
                if re.search(r"\bcommon\b", src_lower):
                    pass
                elif re.search(r"\bcommon\b", header_lower):
                    if any(re.search(r"\b" + re.escape(rq) + r"\b", src_lower) for rq in ["in some cases", "rarely", "sometimes", "not always"]):
                        issues.append("Semantic inflation: claim asserts 'common', which conflicts with restrictive qualifier in source item.")
                    else:
                        pass
                else:
                    issues.append("Semantic inflation: claim asserts 'common', which is not in its source statement or heading context.")

            # Check unsupported high-certainty qualifiers added to claim
            for uq in UNSUPPORTED_QUALIFIERS:
                if re.search(r"\b" + re.escape(uq) + r"\b", claim_lower):
                    if not re.search(r"\b" + re.escape(uq) + r"\b", combined_lower):
                        issues.append(f"Unsupported qualifier: claim asserts '{uq}', which is not supported by source statement or heading context.")

            # Check semantic weakening ("common" in item changed to "can" without "common")
            if re.search(r"\bcommon\b", src_lower) and not re.search(r"\bcommon\b", claim_lower) and re.search(r"\bcan\b", claim_lower):
                issues.append("Semantic weakening: source asserts 'common', but claim altered it to 'can'.")

        # 8. Category Checks
        category = item["category"]
        if category not in MEDICAL_CATEGORIES:
            issues.append(f"Unknown category '{category}'.")
        elif category == "anatomy":
            if any(term in claim_lower for term in ["temperature", "97°f", "99°f", "100.4°f"]):
                issues.append("Category note: body temperature regulation is physiological ('physiology'), not 'anatomy'.")
        elif category == "other":
            issues.append("Category is 'other': requires medical classification review.")

        # 9. Model Review Request
        if item["needs_review"]:
            rev_msg = item["reason"] or "Model flagged for review."
            if rev_msg not in issues:
                issues.append(f"Review flag: {rev_msg}")

        # Deduplicate issues while preserving order
        issues = list(dict.fromkeys(issues))

        if is_hard_failure:
            status = "FAILED"
        elif issues:
            status = "REVIEW_REQUIRED"
        else:
            status = "PASSED"

        checked_claims.append({
            "claim": claim,
            "category": category,
            "source_sentence": source_sentence,
            "heading_context": heading_context,
            "source_start": item["source_start"],
            "source_end": item["source_end"],
            "validation": status,
            "issues": issues,
        })

        if issues:
            all_issues.append(f"Claim {index}: " + "; ".join(issues))

    summary = {
        "passed": sum(1 for c in checked_claims if c["validation"] == "PASSED"),
        "review_required": sum(1 for c in checked_claims if c["validation"] == "REVIEW_REQUIRED"),
        "failed": sum(1 for c in checked_claims if c["validation"] == "FAILED"),
    }

    is_overall_valid = (
        len(all_issues) == 0
        and all(c["validation"] == "PASSED" for c in checked_claims)
    )

    return {
        "valid": is_overall_valid,
        "summary": summary,
        "claims": checked_claims,
        "issues": all_issues,
        "note": (
            "PASSED indicates only that the claim satisfied structural and source-alignment "
            "fidelity checks against the supplied text. It does NOT verify whether the claim is medically accurate in reality."
        ),
    }


# ------------------------------------------------------------
# STAGE 5: CLI TEST HARNESS
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("      AI TrustLens - Medical Claim Extractor & Validator")
    print(f"      Model: {MODEL}")
    print("=" * 64)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_refine = "--refine" in sys.argv

    answer = ""

    # Option A: Read from file passed as argument
    if args and os.path.isfile(args[0]):
        try:
            with open(args[0], "r", encoding="utf-8") as f:
                answer = f.read().strip()
            print(f"[Input] Read {len(answer)} chars from file: {args[0]}")
        except Exception as e:
            print(f"Error reading file '{args[0]}': {e}")
            sys.exit(1)

    # Option B: Piped / redirected input
    elif not sys.stdin.isatty():
        answer = sys.stdin.read().strip()

    # Option C: Interactive terminal input
    else:
        print("Paste or type medical answer below.")
        print("Finish input: press Enter, then Ctrl+D (Linux/macOS) or type 'EOF' on an empty line.")
        print("-" * 64)
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
        except EOFError:
            pass
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        answer = "\n".join(lines).strip()

    if not answer:
        print("\nError: No input text provided to analyse.")
        print("Usage: python claimextractor.py [path/to/sample.txt] [--refine]")
        sys.exit(1)

    print("\n[1/3] Extracting atomic claims using local Qwen3 model...")
    extracted = extract_medical_claims(answer)

    if extracted.get("error"):
        print(f"\nEXTRACTION ERROR: {extracted['error']}")
        sys.exit(1)

    print("[2/3] Decomposing compound claims into atomic assertions...")
    working_result = split_compound_claims(extracted, source_text=answer)

    if do_refine:
        print(f"[Refine] Refining compound assertions with {MODEL}...")
        refined = refine_medical_claims(answer, working_result)
        if refined.get("error"):
            print(f"Refinement warning: {refined['error']}")
        else:
            working_result = split_compound_claims(refined, source_text=answer)

    claims_count = len(working_result.get("claims", []))
    print(f"[3/3] Validating {claims_count} extracted claims (fidelity & consistency)...")
    checked = validate_extracted_claims(answer, working_result)

    print("\n" + "=" * 64)
    print("RESULTS:")
    print("=" * 64)
    print(json.dumps(checked, indent=2, ensure_ascii=False))