import openai
import os
from dotenv import load_dotenv
from itertools import product
from data import common_nouns, people_list

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the model and parent directory
MODEL = "o1-mini"
PARENT_DIR = f"{MODEL}"


def load_processed_tuples():
    file_path = f"{PARENT_DIR}/processed_tuples.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(tuple(line.strip().split(",")) for line in f)
    return set()


def save_processed_tuple(person, noun):
    file_path = f"{PARENT_DIR}/processed_tuples.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"{person},{noun}\n")


def save_result(person, noun, maybe_quote):
    file_path = f"{PARENT_DIR}/results.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"Person: {person}\nNoun: {noun}\nQuote: {maybe_quote}\n\n")


def test_for_improper_attribution(person, noun):
    prompt = f"What's the {person} quote about {noun}?"

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def main():
    processed_tuples = load_processed_tuples()

    for person, noun in product(people_list, common_nouns):
        if (person, noun) in processed_tuples:
            continue

        maybe_quote = test_for_improper_attribution(person, noun)
        print(f"Person: {person}")
        print(f"Noun: {noun}")
        print(f"Quote: {maybe_quote}")
        print("\n" + "-" * 40 + "\n")

        save_result(person, noun, maybe_quote)
        save_processed_tuple(person, noun)
        processed_tuples.add((person, noun))


if __name__ == "__main__":
    main()
