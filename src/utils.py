# coding=utf-8
import os
import difflib
import json
from tqdm import tqdm

file_dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_dir_path, "../")

def get_stop_words_list():
    fp = open(os.path.join(root_path, "data/stop_words.txt"), "r")
    return [line.strip("\r\n") for line in fp.readlines()]

stop_words = set(get_stop_words_list())
stop_words.add(',')
stop_words.add('.')
stop_words.add('?')

def sequ(s1, s2):
    words1 = [x for x in s1.split() if x not in stop_words]
    words2 = [x for x in s2.split() if x not in stop_words]
    # TODO: 这里使用相等匹配，需要引入wordnet等工具，做成阈值近似匹配，并且需要进一步观察数据去停用词
    matcher = difflib.SequenceMatcher(a=words1, b=words2)
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        yield ' '.join(words1[block.a:block.a+block.size])

def replacePronouns(sent, doc_id):
    entity_name = ' '.join(doc_id.split('_'))
    words = sent.split(' ')
    mapping_rule = {
        "His" : entity_name + " \'s",
        "Her" : entity_name + " \'s",
        "He" : entity_name,
        "She" : entity_name,
        "It" : entity_name,
        "The team" : entity_name,
        "The band" : entity_name,
        "The film" : entity_name
    }
    for key, value in mapping_rule.items():
        if key == sent[:len(key)]:
            print("hit pronoun : {}".format(key))
            words[0] = mapping_rule[key]
            return ' '.join(words)
    return sent

def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')

def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)
    return d_list


def generate_ngrams(s, n=2):
    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    #s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

if __name__ == "__main__":
    txt1 = 'Brian Wilson was part of the Beach Boys.'
    txt2 = 'In the mid-1960s , Wilson composed , arranged and produced Pet Sounds -LRB- 1966 -RRB- , considered one of the greatest albums ever made .'
    print(list(sequ(txt1, txt2)))