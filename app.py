from recommender import recommend_random
from utils import movies
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html',name = 'Nathan',movies=movies['title'].to_list())

@app.route('/recommendation')
def recommendation():
    print(request.args)
    titles = request.args.getlist('input')
    ratings = request.args.getlist('rating')
    user_rating = dict(zip(titles,ratings))
    #recs = recommend_random(movies, user_rating,k=3)
    recs= recommend_test_nmf(movies, user_rating, model, k=5)
    return render_template('recommendation.html',recs=recs)

if __name__=='__main__': 
    app.run(debug=True)

