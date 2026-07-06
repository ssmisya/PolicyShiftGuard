from vllm_guard.training.formatting import build_output_instructions


def resolve_output_instructions(response_format: str) -> str:
    if response_format == "think":
        return build_output_instructions(use_think_tags=True)
    if response_format == "nothink":
        return build_output_instructions(no_reason=True)
    return build_output_instructions(no_reason=False)


def build_prompt(instance: dict, output_instructions: str) -> str:
    if instance.get("question"):
        return instance["question"] + output_instructions
    if instance.get("policy"):
        return instance["policy"] + output_instructions
    raise ValueError(f"Instance missing 'question' or 'policy': {list(instance.keys())}")


def build_llavaguard_prompt(instance: dict, output_instructions: str) -> str:
    return build_prompt(instance, output_instructions)
