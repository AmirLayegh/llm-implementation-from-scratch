from importlib.metadata import version
import tiktoken

#print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPalace.")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

decoded_text = tokenizer.decode(integers)
print(decoded_text)

""" Key takeaways:
- BPE handles the unknown tokens by splitting them into smaller tokens.
- BPE includes the special <|endoftext|> token in the vocabulary.
"""

text2 = "Akwiere Amirhossein Layegh"
integers2 = tokenizer.encode(text2, allowed_special={"<|endoftext|>"})
print(integers2)

for integer in integers2:
    token = tokenizer.decode([integer])
    print(token)

