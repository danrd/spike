import json
from typing import List, Set, Union

import sklearn
# from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


# ORKG evaluator, runs generated and target queries and computes f1 and bleu over bindings
class ORKGWholeQueryEvaluationMetric:
    def __init__(self):
        self.f1 = 0  # each iteration
        self.bleu = 0  # run GT, run GPT3 and result(GT) == result(GPT3) only if GT produces a result
        self.bleu_correct_queries = 0
        self.bleu_wrong_queries = 0
        self.target_none_generated_none = 0
        self.target_none_generated_something = 0
        self.target_something_generated_none = 0
        self.target_something_generated_something = 0


    def __get_bindings(self, orkg_rest_api_results) -> Set:
        if orkg_rest_api_results == None:
            bindings = []

        elif "boolean" in orkg_rest_api_results.keys():
            bindings = [str(orkg_rest_api_results["boolean"])]
            print(bindings)
        else:
            query_bindings = orkg_rest_api_results["results"]["bindings"]
            bindings_values: List[str] = [
                uri_info["value"]
                for binding in query_bindings
                for variable_name, uri_info in binding.items()
            ]
            bindings = set(bindings_values)
            bindings = list(bindings)
        return bindings


    def run_f1_evaluator(
            self, target, generated_sparql_query: str
    ):
        # TODO replace dict to LoggedWholeQueryExperimentSample
        sparql = SPARQLWrapper("https://orkg.org/triplestore")

        prefixes = """
        PREFIX orkgr: <http://orkg.org/orkg/resource/>
        PREFIX orkgc: <http://orkg.org/orkg/class/>
        PREFIX orkgp: <http://orkg.org/orkg/predicate/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        """

        target = (prefixes+target).replace('\n', '')
        generated_sparql_query = (prefixes+generated_sparql_query).replace('\n', '')

        #print("GENERATED_QUERY: ", generated_sparql_query)
        # pass reference query and the generated query to db endpoint, parse the response and update the metric
        # generated_sparql_query = " ".join([self.db_prefixes, generated_sparql_query])
        try:
            sparql.setQuery(f"{generated_sparql_query}")
            sparql.setReturnFormat(JSON)
            results_generated_query = sparql.query().convert()

        except Exception as ex:
            # print(ex)
            results_generated_query = None
            if type(ex).__name__ == "QueryBadFormed":
                print("Generated query malformed", ex)

        try:
            sparql.setQuery(f"{target}")  # sparql.setQuery(f"{sample.target}")
            sparql.setReturnFormat(JSON)
            results_reference_query = sparql.query().convert()

        except Exception as ex:
            # print(ex)
            results_reference_query = None
            if type(ex).__name__ == "QueryBadFormed":
                print("Reference query can't be executed anymore", ex)

        # update the metric scores
        # TODO parse the response from the endpoint
        # returns empty list if None
        reference_bindings = self.__get_bindings(results_reference_query)
        generated_bindings = self.__get_bindings(results_generated_query)


        # f1 computed according to GERBIL https://www.semantic-web-journal.net/system/files/swj1838.pdf
        if len(reference_bindings) > 0 and len(generated_bindings) > 0:
            # compute f1
            self.target_something_generated_something += 1


        elif len(reference_bindings) == 0 and len(generated_bindings) > 0:
            # If the golden answerset is empty but the system
            # responds with any answerset, we set precision, recall and F-measure to 0
            reference_bindings = ['']
            self.target_none_generated_something += 1

        elif len(reference_bindings) == 0  and len(generated_bindings) == 0:
            #If the goden answerset is empty and the system does respond with an empty answer, we set precision, recall and F - measure to 1
            reference_bindings = ['']
            generated_bindings = ['']
            self.target_none_generated_none +=1

        elif len(reference_bindings) > 0 and len(generated_bindings) == 0:
            generated_bindings = ['']
            #If there is a golden answer but the QA system resspons with an empty answerset, we assume the
            # system could not respond. P=0, R=0, F1 =0
            self.target_something_generated_none += 1

        else:
            print(f"Broken logic in the experiment design, {sample}")
            raise ValueError()

        return reference_bindings, generated_bindings



def compute_f1(binarizer, references, predictions):
    binarizer.fit(references)
    binarizer.fit(predictions)
    f1_score = sklearn.metrics.f1_score(binarizer.transform(references),
                        binarizer.transform(predictions),
                        average='macro')
    return f1_score


if __name__ == "__main__":
    binarizer = MultiLabelBinarizer()
    with open('data/evaluation_logs/postprocessed_evaluation_logs/EvaluatedFullQueryPromptExperimentSix.json', "r",
              encoding="utf8") as r:
        logs = json.load(r)
        # logs = experiment_result["experiments"]
        whole_query_metric = ORKGWholeQueryEvaluationMetric()
        f1_cum_samples = 0
        predictions = []
        references = []
        for log in logs[0:-1]:
            ref, pred = whole_query_metric.run_f1_evaluator(
                sample=log, generated_sparql_query=""
            )
            f1_score_sample = compute_f1(binarizer=binarizer,
                                  references=[ref],
                                  predictions=[pred])
            f1_cum_samples += f1_score_sample
            "Sample evaluation: "
            print(ref)
            print(pred)
            print(f1_score_sample)
            predictions.append(pred)
            references.append(ref)
        print(f1_cum_samples)
        f1_score = compute_f1(binarizer=binarizer,
                              references=references,
                              predictions=predictions)

    ...
