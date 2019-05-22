from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from data import get_reviews, get_business, get_user
import random
import pandas as pd
import numpy as np
import math
from geopy import distance

def pivot_ratings(df_reviews):
    """ takes a rating table as input and computes the utility matrix """
    # get business and user id's
    business_ids = df_reviews['business_id'].unique()
    user_ids = df_reviews['user_id'].unique()
    
    #creeer zelf pivot table
    pivot_data = df_reviews.pivot_table('stars', index='business_id', columns='user_id')
    
    return pivot_data

def eucledian_distance(matrix, id1, id2):
    # only take the features that have values for both id1 and id2
    selected_features = matrix.loc[id1].notna() & matrix.loc[id2].notna()
    
    # if no matching features, return NaN
    if not selected_features.any():
        return np.nan
    
    # get the features from the matrix
    features1 = matrix.loc[id1][selected_features]
    features2 = matrix.loc[id2][selected_features]
    
    # compute the distances for the features
    distances = (features1 - features2)**2
    
    # return the absolute sum
    return math.sqrt(distances.abs().sum())


def eucledian_similarity(matrix, id1, id2):
    """Compute eucledian similarity between two rows."""
    # compute distance
    distance = eucledian_distance(matrix, id1, id2)
    
    # if no distance could be computed (no shared features) return a similarity of 0
    if distance is np.nan:
        return 0
    
    # else return similarity
    return 1 / (1 + distance)    


def create_similarity_matrix_euclid(matrix):
    """creates the similarity matrix based on eucledian distance"""
    similarity_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.index, dtype=float)
    yas = list(similarity_matrix.index.values)
    xas = list(similarity_matrix.columns.values)
    
    for y in yas:
        for x in xas:
            similarity_matrix.loc[y, x] = eucledian_similarity(matrix, y, x)
    
    return similarity_matrix
   
def intersect_ratio(set1, set2):
    # bereken verhouding aantal overeenkomende categorieen. 
    len1 = len(set1)
    len2 = len(set1.intersection(set2))
    if len2 == 0:
        return 0
    else:
        return len2 / len1

def afstand(cord_bis, lat, long):
    # bereken afstand in kilomters tussen 2 businesses
    cord_bis = cord_bis
    cord_tar = (lat, long)
    
    return (distance.distance(cord_bis, cord_tar).km)

def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    # maak dict van reviewdataframes per stad, met stad als key en reviewdataframe als value 
    d_reviews = {key: pd.DataFrame(get_reviews(key, n=10000)).set_index('user_id') for key in CITIES}
    if not city:
        city = random.choice(CITIES)
    
    #scenario 1
    if user_id is None and business_id is None:
        # selecteer review dataframe voor stad
        df_reviews = d_reviews[city]
        # check aantal reviews per business
        num_rev = df_reviews.groupby("business_id")["stars"].apply(len)
        # check gemiddelde rating van de businesses
        mean_rev = df_reviews.groupby("business_id")["stars"].apply(np.mean)
        # selecteer top 10  best rated businesses met meer dan 60 reviews
        sc1 = mean_rev[num_rev > 60].sort_values(ascending=False)[:10]
        # zet business_id's van top 10 in lijst
        topbusinessid = sc1.index.tolist()
        
        # maak lijst voor aanbevelingen
        finalrc1 = []
        # check voor elke aanbeveling de bedrijfsgegevens
        for busid in topbusinessid:
            recommends = get_business(city, busid)
            your_keys = ['business_id', 'stars', 'name', 'city', 'address']
            finaldict = { your_key: recommends[your_key] for your_key in your_keys }
            # stop alle bedrijfsgegevens in een dict adv top 10 lijst
            finalrc1.append(finaldict)
        # return top 10 in dict format
        return finalrc1
    
    # scenario 2
    if user_id and business_id is None:
        df_reviews = d_reviews[city]
        # als gebruiker geen reviews heeft gemaakt in die stad, scenario 1
        if user_id not in df_reviews.index:
            num_rev = df_reviews.groupby("business_id")["stars"].apply(len)
            mean_rev = df_reviews.groupby("business_id")["stars"].apply(np.mean)
            sc1 = mean_rev[num_rev > 60].sort_values(ascending=False)[:10]
            topbusinessid = sc1.index.tolist()
        
            finalrc1 = []
            for busid in topbusinessid:
                recommends = get_business(city, busid)
                your_keys = ['business_id', 'stars', 'name', 'city', 'address']
                finaldict = { your_key: recommends[your_key] for your_key in your_keys }
                finalrc1.append(finaldict)
            return finalrc1
        # check vereisten user reviews, anders scenario 1
        # vereisten: gemiddelde rating hoger dan 3 & meer dan 10 reviews
        if df_reviews.groupby('user_id')['stars'].mean().loc[user_id] > 3 and df_reviews.groupby('user_id')['stars'].count().loc[user_id] > 10:
            # maak utility matrix van reviews
            utility_matrix = pivot_ratings(df_reviews)
            # maak utility matrix van reviews            
            similarity = create_similarity_matrix_euclid(utility_matrix)
            #selecteer top 5 rated businesses van user
            top5_business = df_reviews.loc[user_id].sort_values(by='stars', ascending=False)[['business_id', 'stars']]
            #en zit dit in een lijst
            top5 = top5_business.iloc[:, 0].tolist()
            top5 = list(dict.fromkeys(top5))[:5]
            # selecteer de meest similar businesses adv user's top 5
            similarity_top5 = similarity.loc[:, top5]
            
            recommendation_list = []
            #geef per bedrijf de 2 meest vergelijkbare bedrijven, herhaal dit 5x en geef lijst met top10 bedrijven
            for i in top5:
                top2 = similarity_top5.nlargest(2,[i]).index.values.tolist()
                recommendation_list.append(top2)
            
            # maak van list in list een single lijst
            topbusinessid2 = [item for final2 in recommendation_list for item in final2]
            finalrc2 = []
            # maak dict van bedrijfsgegevens adv de aanbevelingen
            for busid in topbusinessid2:
                recommends = get_business(city, busid)
                your_keys = ['business_id', 'stars', 'name', 'city', 'address']
                finaldict = { your_key: recommends[your_key] for your_key in your_keys }
                finalrc2.append(finaldict)
            return finalrc2


        # user niet voldaan aan review eisen, scenario 1    
        else:
            df_reviews = d_reviews[city]
            num_rev = df_reviews.groupby("business_id")["stars"].apply(len)
            mean_rev = df_reviews.groupby("business_id")["stars"].apply(np.mean)
            sc1 = mean_rev[num_rev > 60].sort_values(ascending=False)[:10]
            topbusinessid = sc1.index.tolist()
        
            finalrc1 = []
            for busid in topbusinessid:
                recommends = get_business(city, busid)
                your_keys = ['business_id', 'stars', 'name', 'city', 'address']
                finaldict = { your_key: recommends[your_key] for your_key in your_keys }
                finalrc1.append(finaldict)
            return finalrc1
    
    
    # scenario 3
    if business_id and not user_id:
        # laat businessdataframe voor stad
        df_business = pd.DataFrame(BUSINESSES[city])
        # selecteer alleen benodige kolommen, schoon de data op 
        df_business = df_business[['business_id', 'categories', 'latitude', 'longitude', 'review_count', 'stars']]
        df_business['categories'] = df_business['categories'].map(lambda x: x.lower())
        df_business['categories'] = df_business['categories'].map(lambda x: x.replace(' ',''))
        df_business.categories = df_business.categories.str.split(pat = ",").map(set)
        # selecteer categorieen van business_id
        set1 = df_business[df_business.business_id == business_id]['categories'].values[0]
        # bereken ratio overeenkomst in categorie met andere businesses in stad
        df_business['ratio'] = df_business.apply(lambda x: intersect_ratio(set1, x['categories']), axis=1)
        # bereken coordinaten van business_id
        cord_lat = df_business[df_business.business_id == business_id]['latitude'].values[0] 
        cord_long = df_business[df_business.business_id == business_id]['longitude'].values[0] 
        cord_bis = (cord_lat, cord_long)
        # bereken afstand van business_id tot alle andere businesses in stad
        df_business['distance'] = df_business.apply(lambda x: afstand(cord_bis, x['latitude'], x['longitude']), axis=1)
        # berekend score gebaseerd op stars, aantal reviews en afstand naar andere businesses
        df_business['rating'] = df_business['review_count'] * df_business['stars'] - (df_business['distance'] * 10)
        df_business['rating'] = df_business['rating'] * df_business['ratio']
        # selecteerd top 10 businesses gebaseerd op hierboven berekende score
        topbusinessid3 = df_business.sort_values(by='rating', ascending=False)[:10].business_id.values.tolist()
        # creeer dict van bedrijfsgegevens adv top 10 aanbevelingen hier boven
        finalrc3 = []
        for busid in topbusinessid3:
            recommends = get_business(city, busid)
            your_keys = ['business_id', 'stars', 'name', 'city', 'address']
            finaldict = { your_key: recommends[your_key] for your_key in your_keys }
            finalrc3.append(finaldict)
        return finalrc3

    return random.sample(BUSINESSES[city], n)
