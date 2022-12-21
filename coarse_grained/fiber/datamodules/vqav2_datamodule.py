from ..datasets import VQAv2Dataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class VQAv2DataModule(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)
        self.is_cp = _config["is_cp"]
        self.train_subset_ratio = _config["train_subset_ratio"]
        self.val_subset_ratio = _config["val_subset_ratio"]
        self.test_subset_ratio = _config["test_subset_ratio"]

    @property
    def dataset_cls(self):
        return VQAv2Dataset

    @property
    def dataset_name(self):
        return "vqa"

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()

        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_answers = [l for lll in all_answers for ll in lll for l in ll]
        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]

        self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1

        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            is_cp=self.is_cp,
            subset_ratio=self.train_subset_ratio,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            is_cp=self.is_cp,
            subset_ratio=self.val_subset_ratio,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            is_cp=self.is_cp,
            subset_ratio=self.test_subset_ratio,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
        )