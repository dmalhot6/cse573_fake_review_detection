import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

def get_POS_Tagging(sentence_to_tag):
    total_verb_count = 0
    total_noun_count = 0
    pos_tags = []

    text = nltk.word_tokenize(sentence_to_tag)
    tagged_words_list = (nltk.pos_tag(text))

    for lis in tagged_words_list:
        pos_tags.append(lis[1])

    for tag in pos_tags:
        if tag in ['NOUN', 'NNP', 'NN', 'NUM', 'NNS', 'NP', 'NNPS']:
            total_noun_count += 1
        elif tag in ['VERB', 'VB', 'VBN', 'VBD', 'VBZ', 'VBG', 'VBP']:
            total_verb_count += 1
        else:
            continue

    if total_noun_count >= total_verb_count:
        sentence_to_tag = 'T'
    else:
        sentence_to_tag = 'F'

    return sentence_to_tag
