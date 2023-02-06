import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    model = {}

    # probability that any page in the corpus will be chosen given the current page has links to other pages
    corpus_probability = (1 - damping_factor) / (len(corpus))

    # if current page has no links, choose any page in corpus with equal probability, no damping factor
    if len(corpus[page]) == 0:
        for link in corpus.keys():
            model[link] = 1 / len(corpus)
    else:
        # probability that any given link on current page will be chosen
        link_probability = damping_factor / len(corpus[page])

        # current page itself has corpus probability of being chosen
        model[page] = corpus_probability

        # each link on the page also has corpus probability of being chosen
        for link in corpus[page]:
            model[link] = link_probability + corpus_probability
    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result = {}

    # first sample is any page in the corpus with equal probability
    page = random.choice([*corpus])
    result[page] = 1 / n

    # iterate through remaining n-1 samples, calling transition model on each one
    for _ in range(n - 1):
        model = transition_model(corpus, page, damping_factor)

        # use random.choices for weighted probabilities
        page = random.choices([*model], [*model.values()], k=1)[0]

        # keep track of proportionally how much each randomly chosen page shows up in the sample population
        if page in result:
            result[page] += 1 / n
        else:
            result[page] = 1 / n

    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    result = {}

    # initialize all pageranks to 1/N
    for page in corpus.keys():
        result[page] = 1 / len(corpus)

    # continuous iteration until no pagerank changes by more than 0.001
    while True:
        count = 0
        for page in result:

            # keep track of previous pagerank
            old = result[page]

            # new pagerank value
            result[page] = (1 - damping_factor) / len(corpus)
            summation = 0

            # sum pagerank/numlinks for every page that links to current page
            for parent_page in corpus:
                if page in corpus[parent_page]:
                    summation += result[parent_page] / len(corpus[parent_page])

                # pages with no links are treated as having a link to every page in the corpus including themselves
                elif len(corpus[parent_page]) == 0:
                    summation += result[parent_page] / len(corpus)
            result[page] += damping_factor * summation

            # 0.001 threshold check
            if abs(result[page] - old) < 0.001:
                count += 1
        if count == len(corpus):
            break

    return result


if __name__ == "__main__":
    main()
