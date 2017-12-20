import nltk


def lexical_diversity(text):
    """
    Lexical richness of a text.
    Returns proportion of the total text is made up
    of totally distinct words.
    """
    return len(set(text)) / len(text)


def percentage(count, total):
    """
    What percentage of a text is taken up by
    a specific word?
    """
    return 100 * count / total


def unusual_words(text):
    """
    Returns a sorted set of words in the text that are not in
    the set of common words.
    """
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


def content_fraction(text):
    """
    Find the fraction of words in text
    that are not stopwords.
    """
    stopwords = nltk.corpus.stopwords.words("english")
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


def stress(pron):
    """
    Return a list of stresses for an input pronunciation.
    """
    return [char for phone in pron for char in phone if char.isdigit()]

def tabulate(cfdist, words, categories):
    print("{:16}".format("Category"), end=' ')
    for word in words:
        # All words in distribution
        print("{:>6}".format(word), end=' ')
    print()
    for category in categories:
        # Row heading
        print("{:16}".format(category), end=' ')
        for word in words:
            # Frequencies for each word in that (conditional to
            # the current category)
            print("{:6}".format(cfdist[category][word]), end=' ')
        print()
