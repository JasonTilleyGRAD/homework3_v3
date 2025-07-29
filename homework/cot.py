#Used ChatGPT to convert train.json into message prompts for a CoT model.

from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
      
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an expert unit converter who always explains the reasoning before answering. "
                    "For each conversion:\n"
                    "- Show the conversion factor or formula used\n"
                    "- Do the calculation step-by-step\n"
                    "- Then output only the final float value inside <answer>...</answer> tags, "
                    "with full precision and no units or extra text.\n"
                    "Do not round unless explicitly asked. Always be exact.\n"
                )
            },
            {
                "role": "user",
                "content": "What is the measurement of 3 kg when converted into pounds?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilogram = 2.20462262185 pounds\n"
                    "So, 3 × 2.20462262185 = 6.61386786555\n"
                    "<answer>6.61386786555</answer>"
                )
            },
            {
                "role": "user",
                "content": "How many megabytes is 2 gigabytes?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 gigabyte = 1000 megabytes\n"
                    "So, 2 × 1000 = 2000\n"
                    "<answer>2000.0</answer>"
                )
            },
            {
                "role": "user",
                "content": "What is the conversion from months to hours for 3 units?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 month ≈ 730.484 hours\n"
                    "So, 3 × 730.484 = 2191.452 hours\n"
                    "<answer>2191.452</answer>"
                )
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ]




        tokenized = self.tokenizer.apply_chat_template(messages,  tokenize=False)

        return tokenized


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
