import argparse
from typing import Optional, Sequence

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from vllm_guard.training.cls.pipeline import Qwen2_5_VLBinaryClassifier


class BinaryClassifierInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = Qwen2_5_VLBinaryClassifier.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.no_grad()
    def predict(self, image, question: str) -> dict:
        if isinstance(image, str):
            image = Image.open(image)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = torch.argmax(logits, dim=-1).item()
        return {
            "label": pred_label,
            "probs": probs.cpu().tolist(),
            "decision": "block" if pred_label == 1 else "pass",
            "confidence": probs[pred_label].item(),
        }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Run single-image inference with the canonical CLS model")
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("question", nargs="?", default="Is this image safe?")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model = BinaryClassifierInference(args.model_path)
    result = model.predict(args.image_path, args.question)
    print(f"Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: pass={result['probs'][0]:.2%}, block={result['probs'][1]:.2%}")


if __name__ == "__main__":
    main()
