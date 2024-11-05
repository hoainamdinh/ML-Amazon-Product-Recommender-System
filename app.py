from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load the data and model
train_data = pd.read_csv(r'D:\TAI LIEU HOC TAP\Kì 3\Học máy\MID TERM SUBMISSION\Group 09 Model Deploy\cleaned_data.csv')

# Define the content-based recommendation function
def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        return pd.DataFrame()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'IMGURL', 'Rating']]
    return recommended_items_details

# Define the home route with a search bar
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        item_name = request.form['item_name']
        recommendations = content_based_recommendations(train_data, item_name, top_n=10)
        recommendations_list = recommendations.to_dict(orient='records')

        return render_template('index.html', item_name=item_name, recommendations=recommendations_list)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


    # 5. Deploy the application:
    # gcloud app deploy

    # Note: Make sure to replace [YOUR_PROJECT_ID] with your actual Google Cloud project ID.