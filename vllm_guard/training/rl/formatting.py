VALID_RESPONSE_FORMATS = {"think", "nothink"}


THINK_OUTPUT_INSTRUCTIONS = (
    "\n\n"
    "Based on these policies, determine if the image violates any policy.\n"
    "\n"
    "Before the final answer, write a concise <think> section that does all of the following:\n"
    "1. Mention the concrete visual evidence in the image.\n"
    "2. Identify the relevant policy/category requirement.\n"
    "3. Connect the visual evidence to the policy and decide safe vs unsafe.\n"
    "Keep the <think> section short and policy-grounded.\n"
    "\n"
    "Answer in this exact format (no extra text):\n"
    "If content violates a policy: <think>brief reason</think> true | <two-digit Risk Category ID>\n"
    "If content is safe: <think>brief reason</think> false"
)


NO_THINK_OUTPUT_INSTRUCTIONS = (
    "\n\n"
    "Based on these policies, determine if the image violates any policy.\n"
    "\n"
    "Answer in this exact format (no extra text):\n"
    "If content violates a policy: true | <two-digit Risk Category ID>\n"
    "If content is safe: false"
)


def normalize_response_format(response_format: str) -> str:
    normalized = str(response_format or "think").strip().lower()
    if normalized not in VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"Unsupported RL response format: {response_format!r}. "
            f"Expected one of {sorted(VALID_RESPONSE_FORMATS)}"
        )
    return normalized


def get_output_instructions(response_format: str) -> str:
    normalized = normalize_response_format(response_format)
    if normalized == "nothink":
        return NO_THINK_OUTPUT_INSTRUCTIONS
    return THINK_OUTPUT_INSTRUCTIONS
