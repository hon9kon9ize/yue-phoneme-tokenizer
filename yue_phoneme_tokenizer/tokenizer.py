import re
from typing import List
from dataclasses import dataclass
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
    input_ids: List[int]
        A list of phoneme token ids.
    word2ph : List[int]
        A list mapping word indices to phoneme indices.
    """

    input_ids: List[int]
    word2ph: List[int]


class PhonemeTokenizer:
    """
    A tokenizer class for converting text into phonemes.

    Attributes:
        punctuation (set): A set of punctuation characters to be considered during tokenization.
        phoneme_dict (List[str]): A list of phonemes used for tokenization.

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

    def __init__(self, phoneme_dict: List[str], return_punctuation: bool = True):
        self.phoneme_dict = phoneme_dict
        self.return_punctuation = return_punctuation

        if self.return_punctuation:
            orig_dict_len = len(self.phoneme_dict)
            # expand the phoneme dictionary to include punctuation
            self.phoneme_dict = {
                **self.phoneme_dict,
                **{p: i + orig_dict_len for i, p in enumerate(self.punctuation)},
            }

        dictionary_pattern = "|".join(map(re.escape, self.phoneme_dict.keys()))
        self._phone_markup_regex = (
            r"([^\{\}]+)\{\s*("
            + f"(?:{dictionary_pattern})(?:\s+(?:{dictionary_pattern}))*"
            + r")\s*\}"
        )

        if len(self.phoneme_dict) != len(set(self.phoneme_dict)):
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
        input_ids = [
            self.phoneme_dict[token] + 1 for token in tokens
        ]  # +1 for UNK token

        return PhonemeTokenizerEncodedOutput(input_ids=input_ids, word2ph=word2ph)

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the phoneme vocabulary.

        Returns:
            int: The size of the phoneme vocabulary.
        """
        return len(self.phoneme_dict) + 1

    def get_vocab(self):
        """
        Returns the phoneme vocabulary.

        Returns:
            dict: A dictionary mapping phoneme tokens to their corresponding IDs.
        """
        return [self.unk_token] + [
            key for key in self.phoneme_dict.keys()
        ]  # prepend UNK token
