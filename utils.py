import spacy
from prettytable import PrettyTable
from leia.leia import SentimentIntensityAnalyzer as SentimentPT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SentimentEN



def pos_tagger(text, lang, only_keeping_pos):
    pos_desc = {
        "ADJ"  : "adjective",
        "ADP"  : "adposition",
        "ADV"  : "adverb",              
        "AUX"  : "auxiliary",
        "CONJ" : "conjunction",         
        "CCONJ": "coordinating conjunction",
        "DET"  : "determiner",
        "INTJ" : "interjection",
        "NOUN" : "noun",
        "NUM"  : "numeral",
        "PART" : "particle",
        "PRON" : "pronoun",
        "PROPN": "proper noun",
        "PUNCT": "punctuation",
        "SCONJ": "subordinating conjunction",
        "SYM"  : "symbol",
        "VERB" : "verb",
        "X"    : "other",
        "SPACE": "space"
    }
    
    
    nlp_en = spacy.load("en_core_web_sm")
    nlp_pt = spacy.load('pt_core_news_md')
    
    
    if lang=="PT":
        doc = nlp_pt(text)
    else:
        doc = nlp_en(text)
    
    table = PrettyTable(["Word", "POS", "POS Description"])
    for token in doc:
        if token.pos_ in only_keeping_pos:
            table.add_row([token.lemma_, token.pos_, pos_desc[token.pos_]])
    return table


def sentiment_analyzer(text, lang):
    if lang == "PT":
        s = SentimentPT()
    else:
        s = SentimentEN()
        
    sentiment_dict = s.polarity_scores(text)
    
    if sentiment_dict["compound"] >= 0.05:
        sentiment = "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        sentiment = "Negative"
    else :
        sentiment = "Neutral"
    
    return sentiment, sentiment_dict