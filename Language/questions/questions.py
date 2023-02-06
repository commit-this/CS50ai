import nltk
import sys
import os
import string
import numpy

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = os.path.join(directory, filename)
            with open(f) as reader:
                corpus[filename] = reader.read()
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.tokenize.word_tokenize(document.lower())
    filtered = []
    for token in tokens:
        if token not in string.punctuation and token not in nltk.corpus.stopwords.words("english"):
            filtered.append(token)
    return filtered


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # store unique words in a set and then iterate over them to get idf values
    # store idf values in dictionary
    words = set()
    idfs = {}
    for doc in documents:
        for word in documents[doc]:
            words.add(word)
    for word in words:
        count = 0
        for doc in documents:
            if word in documents[doc]:
                count += 1
        idfs[word] = numpy.log(len(documents) / count)
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    rankings = {}
    # iterate over all words in query, check if they are in the document
    # then update dictionary with tf-idf value
    for word in query:
        for file in files:
            if word in files[file]:
                if file in rankings:
                    rankings[file] += files[file].count(word) * idfs[word]
                else:
                    rankings[file] = files[file].count(word) * idfs[word]
    sorted_rankings = sorted(rankings, key=rankings.get, reverse=True)
    return sorted_rankings[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rankings = {}
    # iterate over sentences, keep track of matching word measure and query term density
    # then update dictionary with tuple value
    for sentence in sentences:
        measure = 0
        hits = 0
        for word in query:
            if word in sentences[sentence]:
                measure += idfs[word]
                hits += 1
        density = hits / len(sentence)
        rankings[sentence] = (measure, density)

    # sort sentences by both tuple values measure and density, return top n results
    sorted_rankings = sorted(rankings, key=lambda x: (rankings[x][0], rankings[x][1]), reverse=True)
    return sorted_rankings[:n]


if __name__ == "__main__":
    main()
