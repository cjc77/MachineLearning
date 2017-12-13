

def lexical_diversity(text):
    """
    Lexical richness of a text.
    Returns how much of the total text is made up
    of totally distinct words.
    """
    return len(set(text)) / len(text)

def percentage(count, total):
    """
    What percentage of a text is taken up by
    a specific word?
    """
    return 100 * count / total
