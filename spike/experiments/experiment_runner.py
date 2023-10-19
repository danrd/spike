import json
import logging
from typing import List, Tuple

from spike.llm_runner.chat_gpt_runner import ChatGPTRunner
from spike.SciQA import SciQA
from spike.similarity import rank


class SPARQLQueryGenerationRunner:
    def __init__(self, max_new_tokens):
        self.results = []
        self.llm_runner = ChatGPTRunner(max_new_tokens=max_new_tokens)
        self.sciqa = SciQA()

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
        logger = logging.getLogger('sparqling_with_subgraph')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('logs_test_set_results_experiment_with_example_and_subgraph_2_50_35.log')
        logger.addHandler(fh)
        dataset = self.__load_data(path_to_dataset_with_subgraph)
        for i, sample in enumerate(dataset[195:]):
            print(i)
            instruction = (
                "Instruction",
                "Generate  SPARQL query for the given question. Pay attention to the syntax and namespaces in the example query "
                "and use the structure of the provided knowledge graph while constructing the query. "
                "If question is similar to the example question, then use corresponding SPARQL query as a template for SPARQL query generation. "
                "Do only generate the query, nothing else."
            )
            question = (
                "Question",
                question_string := sample["question"]
            )

            train_example = rank(
                question_string,
                self.sciqa.train.entries,
                top_n = 1,
                get_utterance = lambda entry: entry.utterance,
                threshold = 0
            )[0]

            question_example = (
                'Question example',
                train_example.utterance
            )
            example = (
                "SPARQL query, which corresponds to the Question example",
                train_example.query
            )

            knowledge_graph = (
                "Knowledge graph",
                sample["subgraph"]
            )
            prompt = self.__compose_prompt(prompt_elements=[instruction,
                                           question,
                                           question_example,
                                           example,
                                           knowledge_graph])

            if self.llm_runner.is_prompt_too_long(prompt):
                print("Subgraph length: ", len(sample["subgraph"]))
                knowledge_graph = (
                                      "Knowledge graph",
                                      sample["subgraph"][0:5000]
                                  )
                prompt = self.__compose_prompt(prompt_elements=[instruction,
                                                                question,
                                                                question_example,
                                                                example,
                                                                knowledge_graph])

            print(prompt, flush=True)
            result = self.__call_llm_runner(prompt)
            print(result, flush=True)
            #TODO postprocessing the query
            sample["llm_generated_query"] = result
            logger.info(sample)
            ...
        return dataset


if __name__ == "__main__":
    qg = SPARQLQueryGenerationRunner(max_new_tokens=500)
    generated_queries = qg.process_task(path_to_dataset_with_subgraph="assets/final_questions_with_subgraphs2_50_35.json")
    with open("assets/test_set_results_experiment_with_example_and_subgraph_2_50_35.json", 'w+',
              encoding="utf8") as output_file:
        json.dump(generated_queries, output_file, indent=2)




