from collections import defaultdict
import pandas as pd
from flask import Flask
from flask import request,render_template

app = Flask(__name__)
#----------------------------------------------Machine Learning Script for recommendation of smartphones-----------------------------------------------
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

df_cellphones = pd.read_csv("cellphones data.csv")
df_ratings = pd.read_csv("cellphones ratings.csv")
df_users = pd.read_csv("cellphones users.csv")
df_ratings['rating'] = df_ratings['rating'].replace([18],9)

# Merge the data into a single dataframe
df = pd.merge(df_cellphones,df_ratings, on='cellphone_id')
df.drop_duplicates()

# Step 1 - Data Import & Preparation
# load dataset from pandas data frame , we use load_from_df() method
# also need Reader object and specify rating_scal parameter
# the data frame must have three columns , user id , items id and ratings

data = Dataset.load_from_df(df[['user_id', 'cellphone_id', 'rating']],Reader(rating_scale=(0, 10)))

def get_top_n(predictions,n=5):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    #uid = user id , iid =item id  est == estimated rating
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items(): 
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n


def get_title_from_index(index):
    return (df_cellphones[df_cellphones.index == index]["model"].values[0])

# First train KNNWithMeans algorithm on the dataset.
trainset = data.build_full_trainset()
algo = KNNWithMeans(sim_options={'name': 'cosine'}, verbose=False)
# using fit method which will train the algorithm on trainset
algo.fit(trainset)
# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
"""
top_n=get_top_n(predictions,n=5)
id=8 #input
for uid, user_ratings in top_n.items():
    if uid==id:
         print("User id : "+str(id))
         for u in user_ratings:
             print(get_title_from_index(u[0]))

def getRecommendations(id):
    top_n=get_top_n(predictions,n=5)
    for uid, user_ratings in top_n.items():
        if uid==id:
            print("User id : "+str(id))
            my_list=[]
            for u in user_ratings:
                my_list.append(get_title_from_index(u[0]))
            return my_list

print(getRecommendations(8))
"""

#----------------------------------------------------------------------------------------------------------------------------------------------------
def getRecommendations(id):
    try:
        top_n=get_top_n(predictions,n=5)
        for uid, user_ratings in top_n.items():
            if uid==id:
                my_list=[]
                for u in user_ratings:
                    my_list.append(get_title_from_index(u[0]))
                return my_list
    except ValueError:
        return "invalid input"
lol=8

@app.route("/") # app decorator
def index(): 
    id = request.args.get("id", "")
    if id:
        list=getRecommendations(int(id))
        a=list[0]
        b=list[1]
        c=list[2]
        d=list[3]
        e=list[4]
        rec1,rec2,rec3,rec4,rec5 = a,b,c,d,e
    else:
        rec1,rec2,rec3,rec4,rec5 = "","","","",""
    return (
        "Smartphone Recommendation System"
        + """<form action="" method="get">
                User's id: <input type="text" name="id">
                <input type="submit" value="Enter">
            </form>"""
        + "Recommended for you: "
        + rec1 + " , "
        + rec2 + " , "
        + rec3 + " , "
        + rec4 + " , "
        + rec5 + " "
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)