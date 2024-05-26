from nltk.probability import FreqDist
from nltk.util import ngrams
import math

def generate_language_models(documents):
    unigram_models = {}
    bigram_models = {}
    trigram_models = {}

    for class_name, class_documents in documents.items():
        unigram_freq_dist = FreqDist()
        bigram_freq_dist = FreqDist()
        trigram_freq_dist = FreqDist()

        for document in class_documents:
            unigram_freq_dist.update(document)
            bigrams = list(ngrams(document, 2))
            trigrams = list(ngrams(document, 3))

            bigram_freq_dist.update(bigrams)
            trigram_freq_dist.update(trigrams)

        unigram_models[class_name] = unigram_freq_dist
        bigram_models[class_name] = bigram_freq_dist
        trigram_models[class_name] = trigram_freq_dist

    return unigram_models, bigram_models, trigram_models


def calculate_perplexity(model, document):
    total_log_probability = 0
    total_tokens = len(document)

    for word in document:
        if word in model:
            total_log_probability += math.log(model[word])
        else:
            total_log_probability += math.log(1 / total_tokens)

    perplexity = math.exp(-total_log_probability / total_tokens)
    return perplexity


def calculate_perplexity_with_laplace_smoothing(model, document, vocabulary_size):
    total_log_probability = 0
    total_tokens = len(document)

    for word in document:
        if word in model:
            total_log_probability += math.log((model[word] + 1) / (total_tokens + vocabulary_size))
        else:
            total_log_probability += math.log(1 / (total_tokens + vocabulary_size))

    perplexity = math.exp(-total_log_probability / total_tokens)
    return perplexity


def classify_documents(documents, unigram_models, use_smoothing=False):
    classifications = []

    for document in documents:
        min_perplexity = float('inf')
        predicted_class = None

        for class_name, model in unigram_models.items():
            if use_smoothing:
                perplexity = calculate_perplexity_with_laplace_smoothing(model, document, len(model))
            else:
                perplexity = calculate_perplexity(model, document)

            if perplexity < min_perplexity:
                min_perplexity = perplexity
                predicted_class = class_name

        classifications.append((document, predicted_class))

    return classifications