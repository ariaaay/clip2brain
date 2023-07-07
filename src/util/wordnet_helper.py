# Wordnet helper functions to extract synsets and synsets meanings
# Aria Wang
# 09/20/2018


def get_wn_synsets(lemma):
    """Get all synsets for a word, return a list of [wordnet_label,definition, hypernym_string]
    for all synsets returned."""
    from nltk.corpus import wordnet as wn

    synsets = wn.synsets(lemma)
    out = []
    for s in synsets:
        print(s)
        # if not '.n.' in s.name(): continue # only verbs!
        hyp = ""
        for ii, ss in enumerate(s.hypernym_paths()):
            try:
                hyp += repr([hn.name() for hn in ss]) + "\n"
            except:
                hyp += "FAILED for %dth hypernym\n" % ii
        out.append(dict(synset=s.name(), definition=s.definition(), hypernyms=hyp))
    return out


def get_wn_meaning(lemma):
    """get meaning of a word using wordNet labels"""
    from nltk.corpus import wordnet as wn

    return wn.synset(lemma).definition()


def print_synset_definition(synset):
    """print WordNet Definition of a specific synset"""
    from nltk.corpus import wordnet as wn

    s = wn.synset(synset)

    return s.definition()
