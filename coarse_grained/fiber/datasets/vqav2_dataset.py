import numpy as np
from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", is_cp=False, subset_ratio=1, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        cp_str = "cp" if is_cp else ""

        if split == "train":
            names = [f"vqa{cp_str}v2_train", f"vqa{cp_str}v2_val"]
        elif split == "val":
            names = [f"vqa{cp_str}v2_val"]
        elif split == "test":
            names = [f"vqa{cp_str}v2_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

        original_size = len(self.index_mapper)
        if subset_ratio == 1:
            self.original_idxs = np.arange(original_size)
        else:
            subset_size = int(subset_ratio * original_size)
            self.original_idxs = np.random.choice(original_size, subset_size, replace=False)

    def __len__(self):
        return len(self.original_idxs)

    def __getitem__(self, idx):
        original_idx = self.original_idxs[idx]
        image_tensor = self.get_image(original_idx)["image"]
        text = self.get_text(original_idx)["text"]

        index, question_index = self.index_mapper[original_idx]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }
