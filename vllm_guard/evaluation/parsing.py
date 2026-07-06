import re


def _parse_guardreasoner_decision_value(value: str) -> str | None:
    value = value.lower().strip().strip(".,;")
    if value in ("unharmful", "safe", "false", "nonharmful", "not_harmful", "unrestricted", "allowed", "permitted", "harmless"):
        return "safe"
    if value in ("harmful", "unsafe", "true", "untrue", "violated", "violation", "unallowed", "disallowed", "not_allowed"):
        return "unsafe"
    # Keep substring checks after exact unharmful handling.
    if "unharmful" in value:
        return "safe"
    if "unallowed" in value or "not allowed" in value:
        return "unsafe"
    if "harmful" in value:
        return "unsafe"
    return None


def _parse_guardreasoner_result(result_text: str) -> str | None:
    # GuardReasoner-VL judges a request/response pair. In our image-guardrail
    # prompt, Request corresponds to the image/content to classify; Response is
    # often the assistant's harmless evaluation text, so parsing Response first
    # turns many unsafe images into false safe predictions.
    for field in ("request", "result", "response"):
        match = re.search(rf"{field}\s*:\s*([\w-]+)", result_text)
        if not match:
            continue
        decision = _parse_guardreasoner_decision_value(match.group(1))
        if decision:
            return decision
    return None


def strip_thinking(text: str) -> str:
    if "</think>" in text and "<think>" not in text.split("</think>")[0]:
        return text.split("</think>")[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _decision_text(response: str, response_format: str) -> str | None:
    """Return the answer span used for parsing, or None if think output is incomplete."""
    if response_format != "think":
        return strip_thinking(response)

    if "</think>" not in response:
        return None
    answer = response.rsplit("</think>", 1)[-1].strip()
    if not answer:
        return None
    return answer


def parse_response(response: str, model_type: str = "vllm", response_format: str = "reasoned") -> str:
    if response_format != "think":
        # Some GuardReasoner outputs leave <think> unclosed but still emit a
        # final <result> tag. Parse that tag from raw text before stripping
        # thinking; otherwise strip_thinking() would discard the verdict.
        raw_result_match = re.search(r"<result>\s*(.*?)\s*</result>", response.lower(), re.DOTALL)
        if raw_result_match:
            decision = _parse_guardreasoner_result(raw_result_match.group(1).strip())
            if decision:
                return decision

    parsed_text = _decision_text(response, response_format)
    if parsed_text is None:
        return "invalid"
    raw_text = parsed_text.lower().strip()
    text = parsed_text.lower().strip()

    # GuardReasoner-style outputs are only expected outside the strict think protocol.
    result_match = re.search(r"<result>\s*(.*?)\s*</result>", text, re.DOTALL)
    if result_match:
        result_text = result_match.group(1).strip()
        decision = _parse_guardreasoner_result(result_text)
        if decision:
            return decision

    first_line = text.split("\n")[0].strip()
    if first_line.startswith("true"):
        return "unsafe"
    if first_line.startswith("false"):
        return "safe"
    answer_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if answer_lines:
        last_line = answer_lines[-1]
        if last_line.startswith("true"):
            return "unsafe"
        if last_line.startswith("false"):
            return "safe"
    if re.search(r"(?m)^\s*true\b", raw_text):
        return "unsafe"
    if re.search(r"(?m)^\s*false\b", raw_text):
        return "safe"
    if re.search(r"\b(answer|final answer|decision|final decision)\s*(?:is|:)?\s*true\b", raw_text):
        return "unsafe"
    if re.search(r"\b(answer|final answer|decision|final decision)\s*(?:is|:)?\s*false\b", raw_text):
        return "safe"
    if first_line in ("block",):
        return "unsafe"
    if first_line in ("pass",):
        return "safe"
    if first_line in ("safe", "unsafe"):
        return first_line
    if "unsafe" in first_line and "safe" not in first_line.replace("unsafe", ""):
        return "unsafe"
    if "safe" in first_line and "unsafe" not in first_line:
        return "safe"
    if "unharmful" in text:
        return "safe"
    if "harmful" in text and "unharmful" not in text:
        return "unsafe"
    if "block" in text and "pass" not in text:
        return "unsafe"
    if "pass" in text and "block" not in text:
        return "safe"
    if "unsafe" in text:
        return "unsafe"
    if "safe" in text:
        return "safe"
    return "invalid"


def parse_predicted_categories(response: str, response_format: str = "reasoned") -> list[int]:
    parsed_text = _decision_text(response, response_format)
    if parsed_text is None:
        return []
    raw_text = parsed_text.strip()
    text = raw_text

    result_match = re.search(r"<result>\s*(.*?)\s*</result>", text, flags=re.DOTALL)
    if result_match:
        result_text = result_match.group(1).strip()
        response_match = re.search(r"response:\s*(true|untrue|false)\s*\|?\s*(\d+)?", result_text, re.IGNORECASE)
        if response_match:
            decision = response_match.group(1).lower()
            cat_id = response_match.group(2)
            if decision in ("true", "untrue") and cat_id:
                try:
                    return [int(cat_id)]
                except ValueError:
                    pass
    first_line = text.split("\n")[0].strip().lower()
    if not first_line.startswith("true"):
        fallback = re.search(r"\btrue\s*\|\s*((?:\d+\s*,\s*)*\d+)", raw_text, re.IGNORECASE)
        if not fallback:
            return []
        cat_ids = re.findall(r"\d+", fallback.group(1))
        return [int(c) for c in cat_ids if 1 <= int(c) <= 99]
    parts = first_line.split("|")
    if len(parts) < 2:
        fallback = re.search(r"\btrue\s*\|\s*((?:\d+\s*,\s*)*\d+)", raw_text, re.IGNORECASE)
        if not fallback:
            return []
        cat_ids = re.findall(r"\d+", fallback.group(1))
        return [int(c) for c in cat_ids if 1 <= int(c) <= 99]
    cat_ids = re.findall(r"\d+", parts[1].strip())
    return [int(c) for c in cat_ids if 1 <= int(c) <= 99]
