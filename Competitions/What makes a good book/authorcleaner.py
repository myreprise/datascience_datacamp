import maps as m
import textcleaner as tc
from collections import Counter

def clean_authors(df):
    df['authors_list'] = df['authors'].str.lower().apply(lambda x: x.split("',"))

    data1 = []
    for i, row in df.iterrows():
        authors = row['authors_list']
        data2 = []
        for auth in authors:
            data2.append(clean_author_name(auth))
        data1.append(data2)
    df['authors_list'] = data1
    return df

def clean_author_name(text):
    text = text.replace("'","")
    text = text.replace('"',"")
    text = text.strip()
    return text


def create_author_mapping(df):
    data = []
    for i, row in df.iterrows():
        authors = row['authors_list']
        if len(authors) == 1:
            data.append(clean_author_name(authors[0]))
        else:
            for auth in authors:
                data.append(clean_author_name(auth))

    c = Counter(data)
    map_authors = dict(c)
    print("created author mapping")
    return map_authors



def perform_author_mapping(df, map_authors):
    data = []
    for i, row in df.iterrows():
        authors = row['authors_list']
        total_authors = len(authors)
        if total_authors == 1:
            author = clean_author_name(authors[0])
            data.append(map_authors[author])
        else:
            max_val = 0
            for auth in authors:
                try:
                    author = clean_author_name(auth)
                    val = map_authors[author]
                    if val > max_val:
                        max_val = val
                except:
                    print(f"Value missing: {auth}")
            data.append(max_val)
    df['author_total_book_count'] = data
    print("created author book count")
    return df


def create_unique_author_list(df):
    data = []
    for i, row in df.iterrows():
        author_list = row['authors_list']
        for auth in author_list:
            data.append(auth)
    return list(set(data))


def create_author_popular_map(df, author_list):
    i = 0
    item = {}
    for auth in author_list:
        tmp2 = df[df['authors_list'].apply(lambda x: auth in x)]
        item[auth] = tmp2['popular'].sum()
    return item


def count_total_popular(df, author_popular_map):
    data = []

    for i, row in df.iterrows():
        authors = row['authors_list']
        if len(authors) == 1:
            name = authors[0]
            data.append(author_popular_map[name])
        else:
            max_val = 0
            for auth in authors:
                try:
                    name = auth
                    val = author_popular_map[name]
                    if val > max_val:
                        max_val = val
                except:
                    print(f"Value missing: {auth}")
            data.append(max_val)


    df['total_popular_book_count'] = data
    df['author_other_popular_book_count'] = df['total_popular_book_count'].apply(lambda x: x - 1 if x > 0 else x)
    return df
