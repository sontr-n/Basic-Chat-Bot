import json
from preprocessing import transforming_tokens as transf
import text_similar
import classify

if __name__ == "__main__":
    model = classify.Classify()
    model.read_data("data.json")

    ts = text_similar.TextSimilar()

    # ts.pre_trained_model()
    ts.load_model()
    while True:
        text = input("input: ")
        if text == "q":
            break
        print(ts.answer(text))