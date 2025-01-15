from yue_phoneme_tokenizer.tokenizer import (
    PhonemeTokenizer,
    PhonemeTokenizerOutput,
    PhonemeTokenizerEncodedOutput,
)
import regex as re
from typing import Literal, List

Language = Literal["yue", "en"]


def extract_language_and_text_updated(dialogue: str) -> List[str]:
    # 使用正则表达式匹配<语言>标签和其后的文本
    pattern_language_text = r"<(\S+?)>([^<]+)"
    matches = re.findall(pattern_language_text, dialogue, re.DOTALL)
    # 清理文本：去除两边的空白字符
    matches_cleaned = [(lang.lower(), text.strip()) for lang, text in matches]
    return matches_cleaned


def split_alpha_nonalpha(text, mode=1):
    if mode == 1:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\d\s])(?=[\p{Latin}])|(?<=[\p{Latin}\s])(?=[\u4e00-\u9fff\u3040-\u30FF\d])"
    elif mode == 2:
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\s])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d\s])(?=[\u4e00-\u9fff\u3040-\u30FF])"
    else:
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    return re.split(pattern, text)


def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None
        or (isinstance(item, str) and str(item).isspace())
        or str(item) == ""
    )


def cut_para(text: str):
    splitted_para = re.split("[\n]", text)  # 按段分
    splitted_para = [
        sentence.strip() for sentence in splitted_para if sentence.strip()
    ]  # 删除空字符串
    return splitted_para


def cut_sent(para: str):
    para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def classify_language(text: str, target_languages: list = None) -> str:
    # naive classification
    # count chinese characters
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    # count latin characters
    latin_count = len(re.findall(r"[a-zA-Z]", text))

    if chinese_count > latin_count:
        lang = (
            "yue"
            if "yue" in target_languages
            else "zh" if "zh" in target_languages else None
        )
    else:
        lang = "en" if "en" in target_languages else None

    return lang


def split_by_language(text: str, target_languages: list = None) -> list:
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    sentences = re.split(pattern, text)

    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []

    if target_languages is not None:
        new_sentences = []
        for sentence in sentences:
            new_sentences.extend(split_alpha_nonalpha(sentence))
        sentences = new_sentences

    for sentence in sentences:
        if check_is_none(sentence):
            continue

        lang = classify_language(sentence, target_languages)

        end += text[end:].index(sentence)
        if pre_lang != "" and pre_lang != lang:
            sentences_list.append((text[start:end], pre_lang))
            start = end
        end += len(sentence)
        pre_lang = lang
    sentences_list.append((text[start:], pre_lang))

    return sentences_list


class MultilingualTokenizer(PhonemeTokenizer):
    def __init__(self, languages: List[Language], **kwargs: dict):
        self._tokenizers = {}
        self.languages = languages
        vocab = set()
        sorted(self.languages)  # sort the languages to ensure consistent order

        for lang in languages:
            if lang == "yue":
                from yue_phoneme_tokenizer.yue_tokenizer import (
                    CantonesePhonemeTokenizer,
                )

                self._tokenizers[lang] = CantonesePhonemeTokenizer(**kwargs)
                vocab.update(self._tokenizers[lang].vocab)
            elif lang == "en":
                from yue_phoneme_tokenizer.en_tokenizer import EnglishPhonemeTokenizer

                self._tokenizers[lang] = EnglishPhonemeTokenizer(**kwargs)
                vocab.update(self._tokenizers[lang].vocab)

        # sort the vocab to ensure consistent order
        vocab = sorted(vocab)
        vocab_dict = {p: i for i, p in enumerate(vocab)}

        super().__init__(vocab_dict, **kwargs)

    def _split_by_language(self, text: str) -> List[str]:
        return split_by_language(text, self.languages)

    def _contains_lang_markup(self, text: str) -> bool:
        languages_text = "|".join(self.languages)
        return bool(re.search(rf"<{languages_text}>", text))

    def encode(
        self, text: str, language: Language = None
    ) -> PhonemeTokenizerEncodedOutput:
        output = self.tokenize(text, language)
        tokens = output.tokens
        word2ph = output.word2ph
        token_ids = self.tokens_to_ids(tokens)

        return PhonemeTokenizerEncodedOutput(token_ids=token_ids, word2ph=word2ph)

    def tokenize(
        self, text: str, language: Language | None = None
    ) -> PhonemeTokenizerOutput:
        if language is not None:
            assert language in self.languages, f"Unsupported language: {language}"

            outputs = self._tokenizers[language].tokenize(text)

            return outputs

        if self._contains_lang_markup(text):
            slices = self._language_markup_tokenize(text)
            input_ids = []
            word2ph = []

            for _slice in slices:
                lang, text = _slice
                output = self._tokenizers[lang].tokenize(text)
                input_ids.extend(output.tokens)
                word2ph.extend(output.word2ph)

            return PhonemeTokenizerOutput(input_ids, word2ph)

        else:
            outputs = self._auto_tokenize(text)

        return outputs

    def _language_markup_tokenize(self, text: str) -> PhonemeTokenizerOutput:
        result = []
        for _slice in extract_language_and_text_updated(text):
            result.append(_slice)

        return result

    def _auto_tokenize(self, text: str) -> PhonemeTokenizerOutput:
        sentences = self._split_by_language(text)
        outputs = []

        for sentence in sentences:
            sentence_text, lang = sentence
            output = self._tokenizers[lang].tokenize(sentence_text)
            outputs.append(output)

        tokens = []
        word2ph = []

        for output in outputs:
            tokens.extend(output.tokens)
            word2ph.extend(output.word2ph)

        return PhonemeTokenizerOutput(tokens, word2ph)


if __name__ == "__main__":
    tokenizer = MultilingualTokenizer(["yue", "en"])

    print(tokenizer.vocab_dict)
    print(tokenizer.vocab_size)

    input_text = "我係一個學生，我學緊English。"
    auto_output = tokenizer.tokenize(input_text)

    print(
        auto_output.tokens
    )  # ['ng5', 'o5', 'h6', 'ai6', 'j1', 'at1', 'g3', 'o3', 'h6', 'ok6', 's1', 'aang1', ',', 'ng5', 'o5', 'h6', 'ok6', 'g2', 'an2', ' ', 'IH1', 'NG', 'G', 'L', 'IH0', 'SH', '.']
    print(auto_output.word2ph)  # [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 6, 1]

    markup_input = "<yue>我係一個學生，我學緊 <en>English。"
    markup_output = tokenizer.tokenize(markup_input)

    print("markup_output", markup_output)

    test_encode_output = tokenizer.encode(markup_input)

    print("test_encode_output", test_encode_output)

    assert "".join(auto_output.tokens) == "".join(markup_output.tokens)

    token_outputs = tokenizer.ids_to_tokens(test_encode_output.token_ids)

    print(token_outputs, markup_output.tokens)
    assert token_outputs == markup_output.tokens

    print(tokenizer.encode("素素仲留低咗一个地址记得，系曼谷嘅吞母李府。").token_ids)
