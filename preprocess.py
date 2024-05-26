import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def removeUrlsHtmls(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text, flags=re.MULTILINE)
    return text

def removeSpecialChar(text):
    text = re.sub(r'[\\\'\"\n\r\t]', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'Reuter \x03', '', text, flags=re.MULTILINE)
    return text

def removeRomanNumb(text):
    pattern = r'\b(?:IV|IX|V?I{0,3}|XL|L?X{0,3}|CD|D?C{0,3}|CM|M?C{0,3})+\b'
    text = re.sub(pattern, ' ', text, flags=re.MULTILINE)
    return text

def convert2lower(text): 
    text = text.lower()
    return text 

def removeStopWords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def removePunctuation(text):
    text = re.sub(r'[^\w\s]', '', text, flags=re.MULTILINE)
    return text 

def removeDigit(text):
    text = re.sub(r'\d', '', text, flags=re.MULTILINE)
    return text

def removewhiteSpace(text):
    return  " ".join(text.split())

def remove_rare_words(text, threshold=2):
    def tokenization(text):
        text = text.split(' ')
        text = [txt for txt in text if txt != '']
        text = [txt for txt in text if len(txt) > 1]
        return text

    words = tokenization(text)
    word_freq = Counter(words)

    filtered_words = [word for word in words if word_freq[word] >= threshold]
    return ' '.join(filtered_words)

def tokenization(text):
    text = text.split(' ')
    text = [txt for txt in text if txt != '']
    text = [txt for txt in text if len(txt) > 1]
    return text

def stemming(text):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in text]
    return stemmed_tokens

def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    lemm_text = [word for word in lemm_text if len(word) > 1]
    return lemm_text

def remove_NanValues(text, categories):
    nan_ind_doc = [i for i, tok in enumerate(text) if len(tok) == 0]
    for index in sorted(nan_ind_doc, reverse=True):
        del text[index]
        del categories[index]  
        
    return text, categories

def split_train_test(token_documents, categories, train_ratio=0.8):
    train_samp = int(len(token_documents) * train_ratio)

    train_doc, train_cat = token_documents[:train_samp], categories[:train_samp]
    test_doc, test_cat = token_documents[train_samp:], categories[train_samp:]
    
    print("Train size: ", len(train_doc), '\n', "Test size: ", len(test_doc))
    return train_doc, train_cat, test_doc, test_cat

def list2dict_class(doc, cat):
    earn, acquisitions, grain, crude, money_fx = [], [], [], [], []
    for i in range(len(doc)):
        if cat[i][0] == "earn":
            earn.append(doc[i])
        elif cat[i][0] == "acq":
            acquisitions.append(doc[i])
        elif cat[i][0] == "grain":
            grain.append(doc[i])
        elif cat[i][0] == "crude":
            crude.append(doc[i])
        elif cat[i][0] == "money-fx":
            money_fx.append(doc[i])
        
    cat_doc = {"earn": earn, "acq": acquisitions, "grain": grain, "crude": crude, "money-fx": money_fx}
    return cat_doc