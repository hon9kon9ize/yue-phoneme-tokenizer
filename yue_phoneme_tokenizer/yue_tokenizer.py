from yue_phoneme_tokenizer.tokenizer import PhonemeTokenizer, PhonemeTokenizerOutput
import re
import unicodedata
from ToJyutping import get_jyutping_list
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


class CantonesePhonemeTokenizer(PhonemeTokenizer):
    """
    A tokenizer for converting Cantonese text into phonemes using Jyutping romanization.
    """

    def __init__(self, return_punctuation: bool = True):
        initial_finals = [
            p + str(i)
            for p in YUE_INITIALS.split(" ") + YUE_FINALS.split(" ")
            for i in range(1, 7)
        ]
        phoneme_dict = {phn: i for i, phn in enumerate(initial_finals)}

        super().__init__(phoneme_dict, return_punctuation)

    def _g2p(self, text: str):
        """
        Converts Cantonese text into a list of phonemes and their corresponding word lengths.
        You can change the implementation of this method if you want to use a different Jyutping converter.
        """
        return get_jyutping_list(text)

    def _tokenize(self, text: str) -> PhonemeTokenizerOutput:
        jyutping_list = self._g2p(text)
        ph2word_idx = []
        phones = []
        word_len = len(jyutping_list)

        for i, (word, phone) in enumerate(jyutping_list):
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
                    if final + tone not in self.phoneme_dict:
                        tokens.append("UNK")
                    else:
                        tokens.append(final + tone)
                    phone_len += 1

                word2ph[word_idx] = word2ph[word_idx] + phone_len

            except ValueError:
                tokens.append(phone)

        assert len(tokens) == sum(word2ph), text

        return PhonemeTokenizerOutput(tokens, word2ph)

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


if __name__ == "__main__":
    tokenizer = CantonesePhonemeTokenizer()

    print(tokenizer.get_vocab(), tokenizer.vocab_size)

    input_text = "我唔係廿幾卅個香港人。"
    output = tokenizer.tokenize(input_text)
    token_ids = tokenizer.encode(input_text)

    print(output, token_ids)
