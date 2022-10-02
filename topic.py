import json

import pyLDAvis
import spacy
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud

import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import re

import warnings
import spacy_fastlang

# nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nlp = spacy.load("de_core_news_sm", disable=['parser', 'ner'])
# nlp.add_pipe("language_detector")

warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('result.json') as user_file:
    parsed_json = json.load(user_file)

if __name__ == '__main__':
    df = pd.DataFrame.from_dict(parsed_json, orient='columns')
    df['date'] = pd.to_datetime(df['date'])

    # Removing symbols from Abstracts
    df['Review_Cleaned'] = df.apply(
        lambda row: (re.sub("[^A-Za-z0-9' ]+", " ", str(row['review']))), axis=1)

    # df.info()

    # Tokenization
    df['Review_Cleaned'] = df.apply(
        lambda row: (word_tokenize(str(row['Review_Cleaned']))), axis=1)

    # Running the Stopwords
    stop_words = set(stopwords.words("german"))
    df['Review_Cleaned'] = df.apply(
        lambda row: ([w for w in row['Review_Cleaned'] if w not in stop_words]), axis=1)

    # Lemmatization
    lementize = WordNetLemmatizer()
    df2 = df.apply(lambda row: ([lementize.lemmatize(w) for w in row['Review_Cleaned']]), axis=1)

    # Creating Bigram and Trigram for topic modeling

    bigram = gensim.models.Phrases(df2,
                                   min_count=5,
                                   # This defines the minimum time the words needs to occur to be considered as bigram
                                   threshold=1000)  # The higher threshold fewer phrases.

    trigram = gensim.models.Phrases(bigram[df2], threshold=100)

    # Creating an object
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    # defining the functions stopwords, bigrams and trigrams
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


    def make_bigrams(texts):
        return ([bigram[doc] for doc in texts])


    def make_trigrams(texts):
        return ([trigram[bigram[doc]] for doc in texts])


    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    def findmax(tupla):
        max = 0.0
        for var in tupla:
            if var[1] > max:
                max = var[1]
        return max


    def search(max, tupla):
        for var in tupla:
            if var[1] == max:
                return var[0]


    # Removing of Stop Words
    data_words_nostops = remove_stopwords(df2)

    # Forming of trigrams
    data_words_trigrams = make_trigrams(data_words_nostops)

    # nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    # data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN'])

    # Creating Dictionary and Corpus
    dictionary = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [dictionary.doc2bow(text) for text in data_lemmatized]

    # Building LDA Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=25,  # Identifies the 25 topic trends for transportation
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # doc_lda = lda_model[corpus]

    print(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    num_topics = 25

    for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
        print(str(i) + ": " + topic)
        print()

    for i in range(100):
        print(lda_model[corpus[i]])
        print("\n")

    for i in range(9896):
        result = lda_model[corpus[i]]
        parsed_json[i]["Topic"] = search(findmax(result[0]), result[0])
        parsed_json[i]["Topic_Percentage"] = findmax(result[0])

    with open("final.json", "w") as outfile:
        outfile.write(json.dumps(str(parsed_json)))



    import matplotlib

    matplotlib.use('MACOSX')

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      prefer_horizontal=1,
                      height=330,
                      max_words=200,
                      colormap='flag',
                      collocations=True)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        plt.imshow(cloud.fit_words(dict(lda_model.show_topic(i, 200))))
        plt.gca().set_title('Topic' + str(i), fontdict=dict(size=12))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.suptitle("The Top 25 Research Topic Trend for Transportation Research Part B",
                 y=1.05,
                 fontsize=18,
                 fontweight='bold'
                 )
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


    df.info()
