import openai
import os
from dotenv import load_dotenv
from data import common_nouns, people_list

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o"
PARENT_DIR = f"{MODEL}"


def load_processed_tuples(file_path=f"{PARENT_DIR}/processed_tuples.txt"):
    """
    Load previously processed (person, noun) tuples from a file.

    Args:
        file_path (str): Path to the file containing processed tuples.

    Returns:
        set: A set of tuples containing processed (person, noun) pairs.

    Example:
        >>> load_processed_tuples()
        {('Barack Obama', 'swivel chairs'), ('Bill Gates', 'lampshades')}
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(tuple(line.strip().split(",")) for line in f)
    return set()


def save_processed_tuple(person, noun, file_path=f"{PARENT_DIR}/processed_tuples.txt"):
    """
    Save a processed (person, noun) tuple to a file.

    Args:
        person (str): The name of the person.
        noun (str): The noun associated with the person.
        file_path (str): Path to the file where processed tuples are saved.

    Example:
        >>> save_processed_tuple("Elon Musk", "throw pillows")
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"{person},{noun}\n")


def save_result(person, noun, maybe_quote, file_path=f"{PARENT_DIR}/results.txt"):
    """
    Save the result of a quote attribution test to a file.

    Args:
        person (str): The name of the person.
        noun (str): The noun associated with the person.
        maybe_quote (str): The potential quote attributed to the person.
        file_path (str): Path to the file where results are saved.

    Example:
        >>> save_result("Stephen Hawking", "tape dispensers", "As of my knowledge cutoff in October 2023, there is no widely recognized or documented quote from Stephen Hawking specifically about tape dispensers.")
    """
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

    for person, noun in zip(people_list, common_nouns):
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
