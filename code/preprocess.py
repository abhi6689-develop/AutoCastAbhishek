import json
import copy
import random
import pandas as pd
from datasets import Dataset
from datasets.utils.logging import set_verbosity_error


def convert_questions_to_fid_format(auto_questions):
    fid_questions = []
    for question in auto_questions:
        if question['answer'] is None:
            continue
        fid_questions.append(
            {
                "question_id": str(question['id']),
                "question": question['question'] + ' background: ' + question['background'],
                "answers": [str(question['answer'])],
                'target': str(question['answer']),
                'ctxs': [
                    {
                        "title": "",
                        "text": ""
                    }
                ]
            }
        )

    output_file = "train_data.json"
    with open(output_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(fid_questions, indent=4, ensure_ascii=False) + "\n")
    with open("eval_data.json", "w", encoding='utf-8') as writer:
        writer.write(json.dumps(fid_questions, indent=4, ensure_ascii=False) + "\n")


def load_questions():
    questions_path = '/Users/abhishektiwari/Desktop/sem2/CS542/code/Autocast/data/autocast/autocast_questions.json'
    negations_path = '/Users/abhishektiwari/Desktop/sem2/CS542/code/Autocast/data/autocast/negated_tf_questions.json'
    questions = json.load(open(questions_path))
    negations = json.load(open(negations_path))
    id_to_negation = create_question_dict(negations)
    balanced_questions = []
    for question in questions:
        if question['qtype'] == 't/f':
            negated_question = copy.deepcopy(question)
            negated_question['id'] = 'N' + negated_question['id']
            negated_question['question'] = id_to_negation[question['id']]['negated']
            for t in negated_question['crowd']:
                t['forecast'] = 1 - t['forecast']  
            if question['answer'] is not None: 
                if question['answer'] == 'yes':
                    negated_question['answer'] = 'no'
                else:
                    negated_question['answer'] = 'yes'
            balanced_questions.append(negated_question)
        balanced_questions.append(question)
    return balanced_questions


def separate_questions_by_type(auto_questions):
    multiple_choice_questions = []
    numeric_questions = []
    true_false_questions = []

    for question in auto_questions:
        if question['qtype'] == 'mc':
            multiple_choice_questions.append(question)
        elif question['qtype'] == 'num':
            numeric_questions.append(question)
        elif question['qtype'] == 't/f':
            true_false_questions.append(question)
    return multiple_choice_questions, numeric_questions, true_false_questions


def separate_questions_by_status(auto_questions, time):
    resolved_before_time = []
    resolved_after_time = []
    unresolved_before_time = []
    unresolved_after_time = []
    for question in auto_questions:
        if question['status'] == 'Resolved':
            if question['close_time'] < time:
                resolved_before_time.append(question)
            else:
                resolved_after_time.append(question)
        else:
            if question['publish_time'] < time:
                unresolved_before_time.append(question)
            else:
                unresolved_after_time.append(question)
    return resolved_before_time, resolved_after_time, unresolved_before_time, unresolved_after_time


def create_question_dict(questions):
    id_to_question = {question['id']: question for question in questions}
    return id_to_question

def process_dataset():
    questions = load_questions()
    mc_questions, num_questions, tf_questions = separate_questions_by_type(questions)
    id_to_question = create_question_dict(questions)
    
    return mc_questions, num_questions, tf_questions, id_to_question


def create_csv_files(mc_questions, num_questions, tf_questions):
    tf_df = pd.DataFrame(data={'question': [q['question'] for q in tf_questions], 'label': [1 if q['answer'] == 'yes' else 0 for q in tf_questions]})
    tf_df.to_csv("/Users/abhishektiwari/Desktop/sem2/CS542/code/Autocast/data/intervalQA/tf_questions.csv", index=False)

    mc_data = []
    mc_labels = []

    for question in mc_questions:
        if question['answer'] is None:
            continue
        ans = ord(question['answer']) - ord('A')
        formatted_question = question['question'] + '\n'
        for i in range(0, len(question['choices'])):
            formatted_question += f"{i}.{question['choices'][i]}\n"
        mc_data.append(formatted_question)
        mc_labels.append(min(ans, 5))
        
        shuffle_indices = list(range(0, len(question['choices'])))
        while len(shuffle_indices) > 1 and shuffle_indices[ans] == ans:
            random.shuffle(shuffle_indices)

        formatted_question = question['question'] + '\n'
        for i in range(0, len(question['choices'])):
            formatted_question += f"{i}.{question['choices'][shuffle_indices[i]]}\n"
        mc_data.append(formatted_question)
        temp_label = 5
        for i in range(len(shuffle_indices)):
            if shuffle_indices[i] == ans:
                temp_label = min(i, temp_label)
        mc_labels.append(temp_label)

    mc_df = pd.DataFrame(data={'question': mc_data, 'label': mc_labels})
    mc_df.to_csv("/Users/abhishektiwari/Desktop/sem2/CS542/code/Autocast/data/intervalQA/mc_questions.csv", index=False)

    num_data = []
    num_labels = []

    for question in num_questions:
        if question['answer'] is None:
            continue
        num_data.append(question['question'])
        num_labels.append(float(question['answer']))

    num_df = pd.DataFrame(data={'question': num_data, 'label': num_labels})
    num_df.to_csv("/Users/abhishektiwari/Desktop/sem2/CS542/code/Autocast/data/intervalQA/num_questions.csv", index=False)
