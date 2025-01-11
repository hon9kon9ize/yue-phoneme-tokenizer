from yue_phoneme_tokenizer.tokenizer import (
    PhonemeTokenizer,
    PhonemeTokenizerOutput,
    PhonemeTokenizerEncodedOutput,
)
import re
from dataclasses import dataclass
import unicodedata
from ToJyutping import get_jyutping_list
from typing import List
import pycantonese
import cn2an

YUE_INITIALS = "b c d f g gw h j k kw l m n ng p s t w z"
YUE_FINALS = "aa aai aau aam aan aang aap aat aak ai au am an ang ap at ak e ei eu em eng ep ek i iu im in ing ip it ik o oi ou on ong ot ok oe oeng oek eoi eon eot u ui un ung ut uk yu yun yut M Ng"

chars_rep_map = {
    "更": "更",
    "不": "不",
    "料": "料",
    "聯": "聯",
    "行": "行",
    "利": "利",
    "謢": "護",
    "岀": "出",
    "鎭": "鎮",
    "戯": "戲",
    "旣": "既",
    "立": "立",
    "來": "來",
    "年": "年",
    "㗇": "蝦",
}

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


@dataclass
class WordJyutpingPair:
    word: str | None
    jyutping: str


class CantonesePhonemeTokenizer(PhonemeTokenizer):
    """
    A tokenizer for converting Cantonese text into phonemes using Jyutping romanization.
    """

    def __init__(self, return_punctuation: bool = True):
        initial_finals = [
            p + str(i)
            for p in YUE_INITIALS.split(" ") + YUE_FINALS.split(" ")  # initial + final
            for i in range(1, 7)  # tone 1-6
        ]

        super().__init__(initial_finals, return_punctuation)

    def _g2p(self, text: str):
        """
        Converts Cantonese text into a list of phonemes and their corresponding word lengths.
        You can change the implementation of this method if you want to use a different Jyutping converter.
        """
        word_jyutping = get_jyutping_list(text)
        word_jyutping_paris = []

        for word, jyutping in word_jyutping:
            word_jyutping_paris.append(WordJyutpingPair(word, jyutping))

        return word_jyutping_paris

    def _tokenize_jyutping(
        self, jyutping_list: List[WordJyutpingPair]
    ) -> PhonemeTokenizerOutput:
        ph2word_idx = []
        phones = []
        word_len = len(jyutping_list)

        for i, pair in enumerate(jyutping_list):
            word = pair.word
            phone = pair.jyutping

            if phone is None:
                if not self.return_punctuation:
                    continue

                phone_len = 1
                phones.append(word)
            else:
                phone_len = len(phone.split(" "))
                for ph in phone.split(" "):
                    phones.append(ph)

            ph2word_idx.extend([i] * phone_len)

        tokens = []
        word2ph = [0 for _ in range(word_len)]

        for i, phone in enumerate(phones):
            word_idx = ph2word_idx[i]

            if re.search("^[a-z]{1,6}[1-6]{1}$", phone) is None:
                tokens.append(phone)
                word2ph[word_idx] = 1
                continue

            try:
                jyutping = pycantonese.parse_jyutping(phone)[0]
                phone_len = 0
                tone = jyutping.tone
                initial = jyutping.onset
                final = jyutping.nucleus + jyutping.coda

                if final == "ng" or final == "m":
                    final = final[0].upper() + final[1:]

                if initial != "":
                    tokens.append(initial + tone)
                    phone_len += 1

                if final != "":
                    if final + tone not in self.vocab_dict:
                        tokens.append("UNK")
                    else:
                        tokens.append(final + tone)
                    phone_len += 1

                word2ph[word_idx] = word2ph[word_idx] + phone_len

            except ValueError:
                tokens.append(phone)

        return PhonemeTokenizerOutput(tokens, word2ph)

    def _tokenize(self, text: str) -> PhonemeTokenizerOutput:
        jyutping_list = self._g2p(text)
        temp_output = self._tokenize_jyutping(jyutping_list)
        tokens = temp_output.tokens
        word2ph = temp_output.word2ph

        assert len(tokens) == sum(word2ph), text

        return temp_output

    def _text_normalize(self, text: str):
        text = unicodedata.normalize("NFKC", text)
        pattern = re.compile(
            "|".join(
                re.escape(p) for p in list(chars_rep_map.keys()) + list(rep_map.keys())
            )
        )
        replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

        replaced_text = "".join(
            c
            for c in replaced_text
            if unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH")
            or c in self.punctuation
        )

        return replaced_text

    def _normalize(self, text: str):
        text = cn2an.transform(text, "an2cn")
        text = self._text_normalize(text)

        return text

    def encode_jyutping(self, jyutping: str) -> PhonemeTokenizerEncodedOutput:
        """
        Encodes a given Jyutping string into a PhonemeTokenizerEncodedOutput.

        Args:
            jyutping (str): The Jyutping string to be encoded.

        Returns:
            PhonemeTokenizerEncodedOutput: The encoded output of the Jyutping string.
        """
        jyutping_list = []
        for word in jyutping.split(" "):
            jyutping_list.append(WordJyutpingPair(None, word))

        token_outputs = self._tokenize_jyutping(jyutping_list)
        word2ph = token_outputs.word2ph
        token_ids = self.tokens_to_ids(token_outputs.tokens)

        return PhonemeTokenizerEncodedOutput(token_ids=token_ids, word2ph=word2ph)


if __name__ == "__main__":
    tokenizer = CantonesePhonemeTokenizer()

    print(tokenizer.get_vocab(), tokenizer.vocab_size)

    test_text_input = "我唔係廿幾卅個香港人。"
    test_text_output = tokenizer.tokenize(test_text_input)

    print("test_text_output", test_text_output)

    test_encode_output = tokenizer.encode(test_text_input)

    print("test_encode_output", test_encode_output)

    test_jyutping_input = "ngo5 m4 hai6 jaa6 gei2 saa1 aa6 go3 hoeng1 gong2 jan4 ."
    test_jyutping_output = tokenizer.encode_jyutping(test_jyutping_input)

    print("test_jyutping_output", test_jyutping_output)

    text_token_ids_input = [
        83,
        311,
        442,
        42,
        174,
        48,
        120,
        26,
        224,
        91,
        115,
        120,
        27,
        309,
        37,
        355,
        26,
        332,
        46,
        190,
        455,
    ]
    text_token_ids_output = tokenizer.ids_to_tokens(text_token_ids_input)

    assert text_token_ids_output == test_text_output.tokens
