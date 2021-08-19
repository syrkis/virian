# tokenizer.py
#   tokenizes wikipedia summaries
# by: Noah Syrkis

# imports
import re


# tokenizer function
def tokenizer(summary):
    pattern = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
    tokens = re.findall(pattern, summary.lower())
    return tokens

# dev call
def main():
    for token in tokenizer('12121 2121 8 f w the —— human (body)'):
        print(token.isalpha())

if __name__ == '__main__':
    main()
        
