import json
from typing import List, Tuple

from spike.llm_runner.chat_gpt_runner import ChatGPTRunner


class SPARQLQueryGenerationRunner:
    def __init__(self, max_new_tokens):
        self.results = []
        self.llm_runner = ChatGPTRunner(max_new_tokens=max_new_tokens)

    def __load_data(self, path: str) -> json:
        with open(path, 'r', encoding="utf8") as input_data:
            dataset = json.load(input_data)
        return dataset

    def __compose_prompt(self, prompt_elements: List[Tuple[str, str]]) -> str:
        elements = [": ".join([pe[0], pe[1]]) for pe in prompt_elements]
        prompt = "\n".join(elements)
        return prompt

    def __call_llm_runner(self,
                          prompt: str
                          ):
        return self.llm_runner.run(
            prompt=prompt
        )

    def process_task(self, path_to_dataset_with_subgraph):
        dataset = self.__load_data(path_to_dataset_with_subgraph)
        for sample in dataset["questions"]:
            instruction = (
                "Instruction",
                "Generate  SPARQL query for the given question. Pay attention to the syntax and namespaces in the example query "
                "and use the structure of the provided knowledge graph while constructing the query. "
                "Do only generate the query, nothing else."
            )
            question = (
                "Question",
                sample["question"]["string"]
            )
            example = (
                "SPARQL query example",
                "TODO" # TODO @zeionara example_query selection
            )
            knowledge_graph = (
                "Knowledge graph",
                sample["subgraph"]
            )
            prompt = self.__compose_prompt(prompt_elements=[instruction,
                                           question,
                                           example,
                                           knowledge_graph])
            result = self.__call_llm_runner(prompt)

            #TODO postprocessing the query
            sample["llm_generated_query"] = result
        return dataset


if __name__ == "__main__":
    qg = SPARQLQueryGenerationRunner(max_new_tokens=500)
    generated_queries = qg.process_task(path_to_dataset_with_subgraph="assets/test_questions_with_subgraphs.json")
    with open("assets/SciQA-dataset/test_set_results_experiment_with_example_and_subgraph.json", 'w+',
              encoding="utf8") as output_file:
        json.dump(generated_queries, output_file, indent=2)




