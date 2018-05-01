import json
from preprocessing import transforming_tokens as transf
import text_similar


ts = text_similar.TextSimilar()

# ts.create_dictionary()
# ts.create_model()
ts.load_model()
while True:
    text = input("input: ")
    if text == "q":
        break
    print(ts.predict(text))