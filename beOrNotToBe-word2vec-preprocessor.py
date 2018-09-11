import csv

import nltk.data
from nltk.corpus import gutenberg
from nltk.corpus import reuters

# used for troubleshooting
in_memory_raw_text = """When the modern Olympics began in 1896, the initiators and organizers were looking for a great popularizing event, recalling the ancient glory of Greece! The idea of a marathon race came from Michel Breal, who wanted the event to feature in the first modern Olympic Games in 1896 in Athens. This idea was heavily supported by Pierre de Coubertin, the founder of the modern Olympics, as well as by the Greeks. The Greeks staged a selection race for the Olympic marathon on 10 March 1896 that was won by Charilaos Vasilakos in 3 hours and 18 minutes (with the future winner of the introductory Olympic Games marathon coming in fifth). The winner of the first Olympic Marathon, on 10 April 1896 (a male-only race), was Spyridon "Spyros" Louis, a Greek water-carrier, in 2 hours 58 minutes and 50 seconds. The women's marathon was introduced at the 1984 Summer Olympics (Los Angeles, USA) and was won by Joan Benoit of the United States with a time of 2 hours 24 minutes and 52 seconds. Since the modern games were founded, it has become a tradition for the men's Olympic marathon to be the last event of the athletics calendar, with a finish inside the Olympic stadium, often within hours of, or even incorporated into, the closing ceremonies. The marathon of the 2004 Summer Olympics revived the traditional route from Marathon to Athens, ending at Panathinaiko Stadium, the venue for the 1896 Summer Olympics. The Olympic men's record is 2:06:32."""

be_verb_form = "am are were was is been being be".split()


def load_text():
    nltk.download('gutenberg')
    nltk.download('reuters')
    gutenberg_text = gutenberg.raw()
    reuters_text = reuters.raw()
    return gutenberg_text + reuters_text


def split_sentences(text): return nltk.sent_tokenize(text)


def split_words(text): return nltk.word_tokenize(text)


def split_words_regexp(text):
    pattern = r'''\n'''
    return nltk.regexp_tokenize(text, pattern)


def extract_examples(words):
    # words that are not be verbs
    other_words = []

    # be verbs in sentence
    be_verbs = []

    # split be verbs and other words
    for x in words:
        if x in be_verb_form:
            be_verbs.append(x)
        else:
            other_words.append(x)

    # make one example per be verb in sentence
    examples = []
    for be_verb in be_verbs:
        be_verbs_less_current_verb = be_verbs[:]
        be_verbs_less_current_verb.remove(be_verb)
        other_be_verb_with_other_words = other_words[:]
        for other_be_verb in be_verbs_less_current_verb:
            other_be_verb_with_other_words.append(other_be_verb)
        examples.append((" ".join(other_be_verb_with_other_words), be_verb))
    return examples


def text_to_csv(text):
    sentences = split_sentences(text)
    with open('data.csv', mode='w', newline='', encoding='utf-8') as examples_file:
        example_writer = csv.writer(examples_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for sentence in sentences:
            words = split_words(sentence)
            newExamples = extract_examples(words)
            for example in newExamples:
                list1 = list(example)
                example_writer.writerow(list1)


def main():
    text_to_csv(load_text())


if __name__ == '__main__':
    main()
