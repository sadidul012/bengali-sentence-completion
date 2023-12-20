import pyter
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
import itertools


def bleu(ref, gen):
    """
    calculate pair wise bleu score. uses nltk implementation
    Args:
        ref : a list of reference sentences
        gen : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    """
    ref_bleu = []
    gen_bleu = []
    for x in gen:
        gen_bleu.append(x.split())
    for i, l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu


def ter(ref, gen):
    '''
    Args:
        ref - reference sentences - in a list
        gen - generated sentences - in a list
    Returns:
        averaged TER score over all sentence pairs
    '''
    if len(ref) == 1:
        total_score = pyter.ter(gen[0].split(), ref[0].split())
    else:
        total_score = 0
        for i in range(len(gen)):
            total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
        total_score = total_score / len(gen)
    return total_score


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


# supporting function
def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


# supporting function
def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def rouge_n(reference_sentences, evaluated_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.
    Returns:
      recall rouge score(float)
    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # just returning recall count in rouge, useful for our purpose
    return recall


def score_generated_sentences(generated_sentences, evaluated_sentences):
    return dict(
        bleu=bleu(generated_sentences, evaluated_sentences),
        ter=ter(generated_sentences, evaluated_sentences),
        rouge=rouge_n(generated_sentences, evaluated_sentences)
    )


if __name__ == '__main__':
    train_sentences = ['an apple', 'an orange']
    test_sentences = ['an apple', 'an ant']

    # print("bleu score:", bleu(train_sentences, test_sentences))
    # print("ter score:", ter(train_sentences, test_sentences))
    # print("Rouge score:", rouge_n(train_sentences, test_sentences))
    print(score_generated_sentences(train_sentences, test_sentences))
