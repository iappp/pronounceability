# Roughly determines the pronounceability of a word using the probability of the word as a proxy for pronounceability. This rudimentary language model treats a word as a Markov sequence and then applies the chain rule for probability: P(Y, X1, …, Xn) = P(Y | X1, …, Xn) P(X1, …, Xn)

import numpy as np
import sys


def word_to_character_n_grams(word, n):
    return [word[i:min(len(word), i+n)] for i in range(len(word) - n)]


class Corpus:
    def __init__(self, *wordlists):
        self.n_grams = {}
        self.N = 1 # N=1 for characters, which is probably ideal here
        for wordlist in wordlists:
            with open(wordlist) as f:
                for l in f:
                    word = l.strip().lower()
                    n_grams = word_to_character_n_grams(word, self.N)
                    for idx in range(1, len(n_grams)):
                        if not n_grams[idx] in self.n_grams:
                            self.n_grams[n_grams[idx]] = {}
                        self.n_grams[n_grams[idx]][n_grams[idx-1]] = self.n_grams[n_grams[idx]].get(n_grams[idx-1], 0.) + 1.
        for _, frequency_bucket in self.n_grams.items():
            total = sum(frequency_bucket.values())
            for n_gram, frequency in frequency_bucket.items():
                frequency_bucket[n_gram] = np.log(frequency/total)


    def probability_of_word(self, new_word):
        new_word = new_word.strip().lower()
        new_word_n_grams = word_to_character_n_grams(new_word, self.N)
        if len(new_word_n_grams) == 0:
            return 0.
        default_value = 1./len(self.n_grams.keys())
        p = default_value if new_word_n_grams[0] in self.n_grams else 0.
        q = default_value
        for idx in range(1, len(new_word_n_grams)):
            prev, cur = new_word_n_grams[idx-1], new_word_n_grams[idx]
            if prev not in self.n_grams:
                # Known problem: This doesn't reduce the probability for out-of-vocabulary n-grams; it treats them like they are invisible
                continue
            p += self.n_grams[prev].get(cur, 0.)
            q += 1./len(self.n_grams.get(prev, {}).keys())
        p /= q # normalize for length
        return p


    def relative_pronounceability(self, new_word, other_word='english'):
        "Determines the relative pronounceability of your new word"
        return self.probability_of_word(new_word) - self.probability_of_word(other_word)


    def more_pronounceable_than(self, new_word, other_word):
        return self.relative_pronounceability(new_word, other_word) > 0.


if __name__ == '__main__':
    markov_model = Corpus('/usr/share/dict/american-english') # Works On My Machine™
    if len(sys.argv) > 1:
        print(markov_model.relative_pronounceability(sys.argv[1]))
