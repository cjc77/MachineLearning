def plural(word):
    if word.endswith('y'):
        return word[:-1] + "ies"
    # Word ends in 's' or 'x', or word ends in "sh" or "ch"
    elif word[-1] in "sx" or word[-2:] in ["sh", "ch"]:
        return word + "es"
    elif word.endswith("an"):
        return word[:-2] + "en"
    else:
        return word + 's'
