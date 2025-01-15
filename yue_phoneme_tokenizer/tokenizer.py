import re
from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer
import logging

PUNCTUATION = ["!", "?", "…", ",", ".", "-", " "]
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

logger = logging.getLogger(__name__)


@dataclass
class PhonemeTokenizerOutput:
    """
    A class to represent the output of the Phoneme Tokenizer.

    Attributes:
    ----------
    tokens : List[str]
        A list of phoneme tokens.
    word2ph : List[int]
        A list mapping word indices to phoneme indices.
    """

    tokens: List[str]
    word2ph: List[int]


@dataclass
class PhonemeTokenizerEncodedOutput:
    """
    A class to represent the output of the Phoneme Tokenizer.

    Attributes:
    ----------
    token_ids: List[int]
        A list of phoneme token ids.
    word2ph : List[int]
        A list mapping word indices to phoneme indices.
    """

    token_ids: List[int]
    word2ph: List[int]


class PhonemeTokenizer:
    """
    A tokenizer class for converting text into phonemes.

    Attributes:
        punctuation (set): A set of punctuation characters to be considered during tokenization.
        vocab (List[str]): A list of phonemes used for tokenization.

    Methods:
        normalize(text) -> str:
            Normalizes the input text. This method should be implemented by subclasses.

        tokenize(text):
            Tokenizes the input text by normalizing it and then converting it to phonemes.
            Returns the tokenized output.

        encode(text) -> PhonemeTokenizerEncodedOutput:
            Encodes the input text into a sequence of phoneme token IDs.
            Returns the encoded output.
    """

    punctuation = PUNCTUATION
    unk_token = "UNK"

    def __init__(
        self,
        vocab: List[str],
        return_punctuation: bool = True,
    ):
        self.vocab = (
            [self.unk_token] + vocab if self.unk_token not in vocab else vocab
        )  # prepend UNK token if not present
        self.vocab_dict = {p: i for i, p in enumerate(self.vocab)}
        self.id_vocab_map = {i: p for i, p in enumerate(self.vocab)}
        self.return_punctuation = return_punctuation

        if self.return_punctuation:
            orig_dict_len = len(self.vocab)
            # expand the phoneme dictionary to include punctuation
            self.vocab_dict = {
                **self.vocab_dict,
                **{p: i + orig_dict_len for i, p in enumerate(self.punctuation)},
            }
            self.id_vocab_map = {i: p for p, i in self.vocab_dict.items()}

        dictionary_pattern = "|".join(map(re.escape, self.vocab_dict.keys()))
        self._phone_markup_regex = (
            r"([^\{\}]+)\{\s*("
            + f"(?:{dictionary_pattern})(?:\s+(?:{dictionary_pattern}))*"
            + r")\s*\}"
        )

        if len(self.vocab_dict) != len(set(self.vocab_dict)):
            raise ValueError("Phoneme dictionary has duplicate entries.")

    def _tokenize(self, text: str) -> PhonemeTokenizerOutput:
        """
        Abstract method to tokenize the input text.

        Args:
            text (str): The input text to be converted to phonemes.

        Returns:
            PhonemeTokenizeOutput: The output containing the phonemes.
        """
        raise NotImplementedError

    def _normalize(self, text: str) -> str:
        """
        Abstract method to normalize the input text.

        This method should be implemented by subclasses to provide specific
        normalization logic for the given text.

        Args:
            text (str): The input text to be normalized.

        Returns:
            str: The normalized text.
        """
        raise NotImplementedError

    def _contains_phones_markup(self, text: str) -> bool:
        return bool(re.fullmatch(self._phone_markup_regex, text))

    def tokenize(self, text: str) -> PhonemeTokenizerOutput:
        """
        Tokenizes the given text into phonemes.

        This method first normalizes the input text and then converts it to phonemes
        using the grapheme-to-phoneme (g2p) conversion.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of phonemes corresponding to the input text.
        """
        text = self._normalize(text)
        outputs = self._tokenize(text)

        return outputs

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Converts a list of phoneme tokens to their corresponding IDs.

        Args:
            tokens (List[str]): A list of phoneme tokens.

        Returns:
            List[int]: A list of phoneme token IDs.
        """
        return [self.vocab_dict[token] for token in tokens]

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Converts a list of phoneme token IDs to their corresponding tokens.

        Args:
            token_ids (List[int]): A list of phoneme token IDs.

        Returns:
            List[str]: A list of phoneme tokens.
        """
        return [self.id_vocab_map[i] for i in token_ids]

    def encode(self, text: str) -> PhonemeTokenizerEncodedOutput:
        """
        Encodes the given text into a sequence of phoneme token IDs.

        Args:
            text (str): The input text to be encoded.

        Returns:
            PhonemeTokenizerEncodedOutput: An object containing the encoded input IDs and the word-to-phoneme mapping.
        """
        output = self.tokenize(text)
        tokens = output.tokens
        word2ph = output.word2ph
        token_ids = self.tokens_to_ids(tokens)

        return PhonemeTokenizerEncodedOutput(token_ids=token_ids, word2ph=word2ph)

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the phoneme vocabulary.

        Returns:
            int: The size of the phoneme vocabulary.
        """
        return len(self.vocab_dict)

    def get_vocab(self):
        """
        Returns the phoneme vocabulary.

        Returns:
            dict: A dictionary mapping phoneme tokens to their corresponding IDs.
        """
        return self.vocab  # prepend UNK token
