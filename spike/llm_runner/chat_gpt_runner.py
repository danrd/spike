import os
from typing import Optional
import openai
from openai.openai_object import OpenAIObject
from transformers import GPT2Tokenizer

default_openai_key = os.getenv("OPENAI_API_KEY")
class ChatGPTRunner:
    def __init__(self,
                 max_new_tokens: int,
                 debug: bool = False
                 ):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_new_tokens = max_new_tokens
        self.__debug = debug

    def run(self, prompt: str) -> Optional[str]:
        result = self.__call_chat_gpt(prompt)
        if self.__debug:
            with open("debug.txt", "a", encoding="utf8") as w:
                w.write("#" * 10)
                w.write(" EXAMPLE ")
                w.write("#" * 10)
                w.write("\n")
                w.write("### PROMPT ###\n")
                w.write(prompt)
                w.write("\n\n")
                w.write("### RESULT ###\n")
                w.write(result)
                w.write("\n\n")
        return result

    def is_prompt_too_long(self, prompt: str) -> bool:
        """Returns Whether a given prompt is too long after tokenization.
        """
        tokens = self.tokenizer(prompt)

        print(len(tokens["input_ids"]) + self.max_new_tokens)
        if len(tokens["input_ids"]) + self.max_new_tokens > 4097:
            return True
        else:
            return False

    def __call_chat_gpt(self, prompt: str) -> str:
        api_result = self.__api_call(prompt=prompt)
        return api_result.choices[0].message.content


    def __api_call(self, prompt: str) -> OpenAIObject:
        return openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "user",
                                                       "content": prompt}],
                                            temperature=0.0,
                                            max_tokens=self.max_new_tokens,
                                            top_p=1,
                                            frequency_penalty=0.0,
                                            presence_penalty=0.6,
                                            n=1,
                                            stream=False,
                                            api_key=default_openai_key)
