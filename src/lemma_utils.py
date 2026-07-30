import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in words]
    return ' '.join(lemmatized_words)

print(lemmatize_text("Debugging"))