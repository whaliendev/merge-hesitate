from transformers import AutoTokenizer
def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("codet5-small")
    print(tokenizer.tokenize("Hello, world!"))

if __name__ == "__main__":
    test_tokenizer()
