import json

def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data