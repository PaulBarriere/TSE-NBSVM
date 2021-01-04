'''

These are all the functions use to import and prepare the data

'''
exec(open('TSE-NBSVM/Init.py').read())

def first_line (df):
    l=[]
    for i in range(len(df.reviews)):
        if df.loc[i,'reviews'].startswith('[t]'):
            l.append(i)
    return(l[0])


def flag_title (df, column):
    s = 0
    for i in range(df.shape[0]):
        if df.loc[i, column].startswith('[t]'):
            df.loc[i,'flag_title'] = 1
            s += 1
    return df, s
    
    
def assign_title (df_reviews, df_titles, index_title, start_index):
    df = pd.DataFrame()
    i = start_index
    while (i <= df_reviews.shape[0]) & (df_reviews.loc[i,'index'] < df_titles.loc[index_title+1, 'index']):
        df.loc[i, 'title'] = df_titles.loc[index_title, 'reviews']
        df.loc[i, 'reviews'] = df_reviews.loc[i, 'reviews']
        i += 1
    return(df, i)
    
    
def crea_df (df_reviews, df_titles):
    df_end = pd.DataFrame()
    start = 0
    for j in range(df_titles.shape[0]-1):
        df, i = assign_title(df_reviews, df_titles, j, start)
        df_end = df_end.append(df)
        start = i + 1
    return df_end
    
    
def plus(df, column, i):
    l = 0
    text = df.loc[i, column].split('[')
    for i in range(len(text)):
        if text[i].startswith('+'):
            l += 1
    return (l)
    

def minus(df, column, i):
    l = 0
    text = df.loc[i, column].split('[')
    for i in range(len(text)):
        if text[i].startswith('-'):
            l += 1
    return (l)


def count_plus_minus(df, column):
    for i in range(len(df)):
        df.loc[i, 'Positive_aspect'] = plus(df, column, i)
        df.loc[i, 'Negative_aspect'] = minus (df, column, i)
    return df
    
    
def positive_or_negative_review (df):
    for i in range(df.shape[0]):
        if df.Positive_aspect[i] >= df.Negative_aspect[i]:
            df.loc[i, 'Feeling_of_the_review'] = 1
        else : 
            df.loc[i, 'Feeling_of_the_review'] = -1
    return df


def import_data(nom_fichier):
    df = pd.read_csv('TSE-NBSVM/Data/' + nom_fichier, sep = '\\') 
    df.columns = ['reviews']
    df= df.drop([i for i in range(first_line(df))], axis = 0)
    df = df.reset_index()
    df = df.drop('index', axis = 1)
    a, b = flag_title(df, 'reviews')
    inte = crea_df(a[a.flag_title!=1].reset_index(), a[a.flag_title==1].reset_index())
    final = inte.groupby('title')['reviews'].apply(lambda x: "%s" % ', '.join(x))
    final = pd.DataFrame(final).reset_index()
    final = count_plus_minus(final, 'reviews')
    final = positive_or_negative_review(final)
    print('Number of reviews for this product :' + str(final.shape[0]))
    return final
    

def convert_text_to_lowercase(df, colname):
    df[colname] = df[colname].str.lower()
    return df


def not_regex(pattern):
        return r"((?!{}).)".format(pattern)


def remove_punctuation(df, colname):
    df[colname] = df[colname].str.replace('\n', ' ')
    df[colname] = df[colname].str.replace('\r', ' ')
    alphanumeric_characters_extended = '(\\b[-/]\\b|[a-zA-Z0-9])'
    df[colname] = df[colname].str.replace(not_regex(alphanumeric_characters_extended), ' ')
    return df


def tokenize_sentence(df, colname):
    df[colname] = df[colname].str.split()
    return df


def remove_stop_words(df, colname):
    stop_words = stopwords.words('english')
    df[colname] = df[colname].apply(lambda x: [word for word in x if word not in stop_words])
    return df


def remove_alone_numbers (df, colname):
    chiffres = [str(i) for i in range(10)]
    df[colname] = df[colname].apply(lambda x: [word for word in x if word not in chiffres])
    return df


def reverse_tokenize_sentence(df, colname):
    df[colname] = df[colname].map(lambda word: ' '.join(word))
    return df


def text_cleaning(df, colname):
    """
    Takes in a string of text, then performs the following:
    1. convert text to lowercase
    2. remove punctuation and new line characters '\n'
    3. Tokenize sentences
    4. Remove all stopwords
    5. convert tokenized text to text
    """
    df = (
        df
        .pipe(convert_text_to_lowercase, colname)
        .pipe(remove_punctuation, colname)
        .pipe(tokenize_sentence, colname)
        .pipe(remove_stop_words, colname)
        .pipe(remove_alone_numbers, colname)
        .pipe(reverse_tokenize_sentence, colname)
    )
    return df


def cleaning_df (df, columns):
    for colname in columns : 
        text_cleaning(df, colname)
    return df
    

def data_ready (dico):
    d = {}
    for i in range(len(dico.keys())):
        dd = {}
        dd['x'] = dico[list(dico.keys())[i]].reviews
        dd['y'] = dico[list(dico.keys())[i]].Feeling_of_the_review
        d[i] = dd
    return d
    

def import_all_data (liste_fichier):
    d1 = {}
    for i in range(len(liste_fichier)) :   
        df = import_data(liste_fichier[i])
        d1[i] = cleaning_df(df, df.columns[:2])   
    d = data_ready(d1)
    return list(d[0]['x'].values), list(d[0]['y'].values), list(d[1]['x'].values), list(d[1]['y'].values), list(d[2]['x'].values), list(d[2]['y'].values), list(d[3]['x'].values), list(d[3]['y'].values), list(d[4]['x'].values), list(d[4]['y'].values)
