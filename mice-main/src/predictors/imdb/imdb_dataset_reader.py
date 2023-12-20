from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
import numpy as np
import math

from src.predictors.predictor_utils import clean_text

logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.9


def get_label(p):
    assert "pos" in p or "neg" in p
    return "1" if "pos" in p else "0"


@DatasetReader.register("imdb")
class ImdbDatasetReader(DatasetReader):
    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0  # numpy random seed

    def get_path(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and \
                not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
            path = chain(
                Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
                Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        elif file_path in ['train_split', 'dev_split']:
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
            path = chain(
                Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
                Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
            path_lst = list(path)
            np.random.shuffle(path_lst)
            num_train_strings = math.ceil(
                TRAIN_VAL_SPLIT_RATIO * len(path_lst))
            train_path, path_lst[:num_train_strings]
            val_path = path_lst[num_train_strings:]
            path = train_path if file_path == "train" else val_path
        elif file_path == 'test':
            pos_dir = osp.join(self.TEST_DIR, 'pos')
            neg_dir = osp.join(self.TEST_DIR, 'neg')
            path = chain(
                Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
                Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        elif file_path == "unlabeled":
            unsup_dir = osp.join(self.TRAIN_DIR, 'unsup')
            path = chain(Path(cache_dir.joinpath(unsup_dir)).glob('*.txt'))
        else:
            raise ValueError(f"Invalid option for file_path.")
        return path

    def get_inputs(self, file_path, return_labels=False):
        np.random.seed(self.random_seed)

        path_lst = list(self.get_path(file_path))
        # strings = [None] * len(path_lst)
        strings = []
        # labels = [None] * len(path_lst)
        labels = []
        # for i, p in enumerate(path_lst):
        #     labels[i] = get_label(str(p))
        #     strings[i] = clean_text(p.read_text(),
        #                             special_chars=["<br />", "\t"])
        labels.append(str("1"))
        strings.append("One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.")
        print(strings)
        print(labels)
        if return_labels:
            return strings, labels
        return strings

    @overrides
    def _read(self, file_path):
        np.random.seed(self.random_seed)
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and \
                not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)
        path = self.get_path(file_path)
        for p in path:
            label = get_label(str(p))
            yield self.text_to_instance(
                clean_text(p.read_text(), special_chars=["<br />", "\t"]),
                label)

    def text_to_instance(
            self, string: str, label: str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)