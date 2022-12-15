import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1].split("_")[-1][:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas] if "test" not in split else list(list())
    answer_labels = [a["labels"] for a in answers] if "test" not in split else list(list())
    answer_scores = [a["scores"] for a in answers] if "test" not in split else list(list())
    answers = [[label2ans[l] for l in al] for al in answer_labels] if "test" not in split else list(list())

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/vqacp_v2_train_questions.json", "r") as fp:
        questions_train = json.load(fp)["questions"]
    with open(f"{root}/vqacp_v2_test_questions.json", "r") as fp:
        questions_test = json.load(fp)["questions"]

    with open(f"{root}/vqacp_v2_train_annotations.json", "r") as fp:
        annotations_train = json.load(fp)["annotations"]
    with open(f"{root}/vqacp_v2_test_annotations.json", "r") as fp:
        annotations_test = json.load(fp)["annotations"]

    annotations = dict()

    for split, questions in zip(
        ["train", "test"],
        [
            questions_train,
            questions_test
        ],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(questions):
            _annot[q["image_id"]][q["question_id"]] = [q["question"]]

        annotations[split] = _annot

    all_major_answers = list()

    for split, annots in zip(
        ["train", "test"],
        [annotations_train, annotations_test],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            all_major_answers.append(q["multiple_choice_answer"])

    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())

    for split, annots in zip(
        ["train", "test"],
        [annotations_train, annotations_test],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            answers = q["answers"]
            answer_count = {}
            for answer in answers:
                answer_ = answer["answer"]
                answer_count[answer_] = answer_count.get(answer_, 0) + 1

            labels = []
            scores = []
            for answer in answer_count:
                if answer not in ans2label:
                    continue
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)

            _annot[q["image_id"]][q["question_id"]].append(
                {
                    "labels": labels,
                    "scores": scores,
                }
            )

    for split in ["train", "test"]:
        filtered_annot = dict()
        for ik, iv in annotations[split].items():
            new_q = dict()
            for qk, qv in iv.items():
                if len(qv[1]["labels"]) != 0:
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot

    for split in [
        "train",
        "test"
    ]:
        annot = annotations[split]
        paths = list(glob(f"{root}/{split}/*.jpg"))
        random.shuffle(paths)
        annot_paths = [path for path in paths if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths),
            len(annot_paths),
            len(annot),
        )

        bs = [path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)]

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/vqacpv2_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)