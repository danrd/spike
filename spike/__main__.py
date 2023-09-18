from click import group, argument, option

from openai import ChatCompletion as cc

from .OrkgContext import OrkgContext
from .similarity import compare as compare_strings, rank
from .SciQA import SciQA
from .endpoint_evaluator import ORKGWholeQueryEvaluationMetric


NEW_LINE = '\n'


@group()
def main():
    pass


# @main.command()
# @argument('question', type = str)
# @option('-d', '--dry-run', is_flag = True, help = 'Print generated context and exit')
# @option('-f', '--fresh', is_flag = True, help = 'Don\'t use cached context entries, generate them from scratch')
def ask(question: str, dry_run: bool, fresh: bool):
    context = OrkgContext(fresh = fresh)

    # if dry_run:
    #     # print(context.description)
    #     print(context.cut(question))
    # else:
    examples, graph = context.cut(question)

    string_examples = []

    for example in examples:
        string_examples.append(f'Also I know that for a similar question "{example.utterance}" the correct query is \n```\n{example.query}\n```.')

    # print('\n'.join(string_examples))

    content = f'''
    I have a knowledge graph which includes the following fragment:

    '
    {graph}
    '

    Generate SPARQL query which allows to answer the question "{question}" using this graph

    {NEW_LINE.join(string_examples)}
    Do only generate the query, nothing else.
    '''

    # print(content)

    if not dry_run:
        completion = cc.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {
                    'role': 'user',
                    'content': content
                }
            ]
        )

        # print(completion)
        return completion.choices[0].message.content


@main.command()
@argument('lhs', type = str)
@argument('rhs', type = str)
def compare(lhs: str, rhs: str):
    print(f'The similarity of strings "{lhs}" and "{rhs}" is {compare_strings(lhs, rhs)}')


@main.command()
@option('-n', '--top-n', type = int, default = 3)
def trace(top_n: int):
    # print(rank('foo', ['qux', 'o', 'fo'], top_n = 2))

    sciqa = SciQA()

    train_entries = sciqa.train.entries

    for test_utterance in sciqa.test.utterances[:1]:
        print(f'Test sample: {test_utterance}')
        print(f'Similar train samples: {rank(test_utterance, train_entries, top_n, get_utterance = lambda entry: entry.utterance)}')
        print('')

    # print(len(sciqa.train.utterances))
    # print(len(sciqa.test.utterances))

@main.command()
@argument('data', type = str) # train/test/valid
def evaluate(data:str):
    sciqa = SciQA()
    if data=='train':
        dataset = sciqa.train.entries
    elif data=='test':
        dataset = sciqa.test.entries
    elif data=='valid':
        dataset = sciqa.valid.entries
    eval = ORKGWholeQueryEvaluationMetric()
    for index, element in enumerate(dataset):
        question = element.utterance
        target = element.query 
        res = ask(question=question, dry_run=False, fresh=True)
        eval.run_f1_evaluator(target, res)
        if (index+1)%10 == 0:
            print(f'processed {index} queries')
    return eval.target_something_generated_something

if __name__ == '__main__':
    main()
