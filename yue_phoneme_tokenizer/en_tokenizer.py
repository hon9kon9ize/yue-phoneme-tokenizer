import pickle
import os
import re
from typing import List
from g2p_en import G2p
from transformers import DebertaV2Tokenizer
from yue_phoneme_tokenizer.tokenizer import PhonemeTokenizer, PhonemeTokenizerOutput
import inflect

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
EN_ARPA = "AH0 S AH1 EY2 AE2 EH0 OW2 UH0 NG B G AY0 M AA0 F AO0 ER2 UH1 IY1 AH2 DH IY0 EY1 IH0 K N W IY2 T AA1 ER1 EH2 OY0 UH2 UW1 Z AW2 AW1 V UW2 AA2 ER AW0 UW0 R OW1 EH1 ZH AE0 IH2 IH Y JH P AY1 EY0 OY2 TH HH D ER0 CH AO1 AE1 AO2 OY1 AY2 IH1 OW0 L SH"

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
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
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

post_rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "v": "V",
}


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


class EnglishPhonemeTokenizer(PhonemeTokenizer):
    """
    A tokenizer class for English text that converts text into phonemes and handles various text normalization tasks.
    """

    _inflect = inflect.engine()
    _comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
    _decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
    _pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
    _dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
    _ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
    _number_re = re.compile(r"[0-9]+")

    # List of (regular expression, replacement) pairs for abbreviations:
    _abbreviations = [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ]

    # List of (ipa, lazy ipa) pairs:
    _lazy_ipa = [
        (re.compile("%s" % x[0]), x[1])
        for x in [
            ("r", "ɹ"),
            ("æ", "e"),
            ("ɑ", "a"),
            ("ɔ", "o"),
            ("ð", "z"),
            ("θ", "s"),
            ("ɛ", "e"),
            ("ɪ", "i"),
            ("ʊ", "u"),
            ("ʒ", "ʥ"),
            ("ʤ", "ʥ"),
            ("ˈ", "↓"),
        ]
    ]

    # List of (ipa, lazy ipa2) pairs:
    _lazy_ipa2 = [
        (re.compile("%s" % x[0]), x[1])
        for x in [
            ("r", "ɹ"),
            ("ð", "z"),
            ("θ", "s"),
            ("ʒ", "ʑ"),
            ("ʤ", "dʑ"),
            ("ˈ", "↓"),
        ]
    ]

    # List of (ipa, ipa2) pairs
    _ipa_to_ipa2 = [
        (re.compile("%s" % x[0]), x[1]) for x in [("r", "ɹ"), ("ʤ", "dʒ"), ("ʧ", "tʃ")]
    ]

    def __init__(self, return_punctuation: bool = True):
        phoneme_dict = {phn: i for i, phn in enumerate(EN_ARPA.split(" "))}
        super().__init__(phoneme_dict, return_punctuation)

        self.eng_dict = get_dict()
        self.g2p = G2p()
        self.word_tokenizer = DebertaV2Tokenizer.from_pretrained(
            "microsoft/deberta-v3-large"
        )

    def _expand_dollars(self, m):
        match = m.group(1)
        parts = match.split(".")
        if len(parts) > 2:
            return match + " dollars"  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return "%s %s" % (dollars, dollar_unit)
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s" % (cents, cent_unit)
        else:
            return "zero dollars"

    def _remove_commas(self, m):
        return m.group(1).replace(",", "")

    def _expand_ordinal(self, m):
        return self._inflect.number_to_words(m.group(0))

    def _expand_number(self, m):
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return "two thousand"
            elif num > 2000 and num < 2010:
                return "two thousand " + self._inflect.number_to_words(num % 100)
            elif num % 100 == 0:
                return self._inflect.number_to_words(num // 100) + " hundred"
            else:
                return self._inflect.number_to_words(
                    num, andword="", zero="oh", group=2
                ).replace(", ", " ")
        else:
            return self._inflect.number_to_words(num, andword="")

    def _expand_decimal_point(self, m):
        return m.group(1).replace(".", " point ")

    def _replace_punctuation(self, text):
        pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
        replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

        return replaced_text

    def _normalize_numbers(self, text: str):
        text = re.sub(self._comma_number_re, self._remove_commas, text)
        text = re.sub(self._pounds_re, r"\1 pounds", text)
        text = re.sub(self._dollars_re, self._expand_dollars, text)
        text = re.sub(self._decimal_number_re, self._expand_decimal_point, text)
        text = re.sub(self._ordinal_re, self._expand_ordinal, text)
        text = re.sub(self._number_re, self._expand_number, text)
        return text

    def _normalize(self, text: str):
        text = self._normalize_numbers(text)
        text = self._replace_punctuation(text)
        text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)

        return text

    def _distribute_phone(self, n_phone: int, n_word: int):
        phones_per_word = [0] * n_word
        for _ in range(n_phone):
            min_tasks = min(phones_per_word)
            min_index = phones_per_word.index(min_tasks)
            phones_per_word[min_index] += 1
        return phones_per_word

    def _post_replace_ph(self, ph):
        if ph in post_rep_map.keys():
            ph = rep_map[ph]
        if ph in self.phoneme_dict:
            return ph
        if ph not in self.phoneme_dict:
            ph = self.unk_token
        return ph

    def _text_to_words(self, text: str):
        tokens = self.word_tokenizer.tokenize(text)
        words = []
        for idx, t in enumerate(tokens):
            if t.startswith("▁"):
                words.append([t[1:]])
            else:
                if t in self.punctuation:
                    if idx == len(tokens) - 1:
                        words.append([f"{t}"])
                    else:
                        if (
                            not tokens[idx + 1].startswith("▁")
                            and tokens[idx + 1] not in self.punctuation
                        ):
                            if idx == 0:
                                words.append([])
                            words[-1].append(f"{t}")
                        else:
                            words.append([f"{t}"])
                else:
                    if idx == 0:
                        words.append([])
                    words[-1].append(f"{t}")
        return words

    def _tokenize(self, text: str):
        tokens = []
        phone_len = []
        words = self._text_to_words(text)

        for word in words:
            temp_tokens = []
            if len(word) > 1:
                if "'" in word:
                    word = ["".join(word)]
            for w in word:
                if w in self.punctuation:
                    temp_tokens.append(w)
                    continue
                if w.upper() in self.eng_dict:
                    phns = [
                        ph for ph_list in self.eng_dict[w.upper()] for ph in ph_list
                    ]
                    temp_tokens += [self._post_replace_ph(i) for i in phns]
                else:
                    phone_list = list(filter(lambda p: p != " ", self.g2p(w)))
                    phns = []
                    for ph in phone_list:
                        if ph in self.phoneme_dict:
                            phns.append(ph)
                        else:
                            phns.append(ph)
                    temp_tokens += [self._post_replace_ph(i) for i in phns]
            tokens += temp_tokens
            phone_len.append(len(temp_tokens))

        word2ph = []
        for token, pl in zip(words, phone_len):
            word_len = len(token)

            aaa = self._distribute_phone(pl, word_len)
            word2ph += aaa

        assert len(tokens) == sum(word2ph), text

        return PhonemeTokenizerOutput(tokens, word2ph)


if __name__ == "__main__":
    tokenizer = EnglishPhonemeTokenizer()
    input_text = "I am a student. I got £100 in my pocket."
    output = tokenizer.tokenize(input_text)
    token_ids = tokenizer.encode(input_text)

    print(output, token_ids)
