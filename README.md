# Cantonese Phoneme Tokenizer

Tokenize Cantonese and English text to phonemes.

This library focuses on the grapheme-to-phoneme and tokenization (text → phoneme → token) process, with many parts of the code borrowed from [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We have refactored this component to make it more efficient and portable. Additionally, we have added features to support Cantonese and mixed-language text, enabling seamless handling of Cantonese and English combinations.

The grapheme-to-phoneme (G2P) conversion is powered by the ToJyutping library, ensuring accurate and reliable phoneme generation. This library is particularly useful for TTS training, providing an efficient solution for G2P conversion and tokenization tasks.

## Installation

```bash
pip install git+https://github.com/hon9kon9izer/yue-phoneme-tokenizer.git
```

## Usage

```python
from yue_phoneme_tokenizer import MultilingualTokenizer, CantoneseTokenizer, EnglishTokenizer

tokenizer = MultilingualTokenizer(["yue", "en"])
input_text = "我係一個學生，我學緊 English。"
auto_output = tokenizer.tokenize(input_text)

print(
    auto_output.tokens
)
# ['ng5', 'o5', 'h6', 'ai6', 'j1', 'at1', 'g3', 'o3', 'h6', 'ok6', 's1', 'aang1', ',', 'ng5', 'o5', 'h6', 'ok6', 'g2', 'an2', ' ', 'IH1', 'NG', 'G', 'L', 'IH0', 'SH', '.']
print(auto_output.word2ph)
# [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 6, 1]

markup_input = "<yue>我係一個學生，我學緊<en>English。"
markup_output = tokenizer.tokenize(markup_input)
# [83, 311, 42, 174, 43, 205, 27, 309, 42, 348, 91, 145, 532, 83, 311, 42, 348, 26, 188, 535, 68, 9, 11, 70, 24, 71, 533]

# Cantonese tokenizer
cantonese_tokenizer = CantoneseTokenizer()
cantonese_output = cantonese_tokenizer.tokenize("我係一個學生。")
print(cantonese_output.tokens)
# ['ng5', 'o5', 'h6', 'ai6', 'j1', 'at1', 'g3', 'o3', 'h6', 'ok6', 's1', 'aang1', '.']

# English tokenizer
english_tokenizer = EnglishTokenizer()
english_output = english_tokenizer.tokenize("I am a student.")
print(english_output.tokens)
# ['AY1', 'AE1', 'M', 'AH0', 'S', 'T', 'UW1', 'D', 'AH0', 'N', 'T', '.']
```

## References

- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [Bert-VITS2-Cantonese](https://github.com/hon9kon9ize/Bert-VITS2-Cantonese)