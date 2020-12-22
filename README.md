# Predict Google Play Store App Ratings
#Training Neural Network for Classification

<h4>Installation</h4>

<ul>
<li>Numpy</li>
<li>Matplotlib</li>
<li>scikit-learn</li>
<li>tensorflow 2.1</li>
<li>pandas </li>
<li>nltk</li>
<li>spaCy</li> 
</ul>

<h5>Download</h5>
// stopwords
nltk.download('stopwords') 

<h5>Introduction</h5>

<p>The ability to use services and products on the go has been a major leap in this century. Applications on the Google play store aim to do exactly that. Owing to worldwide accessibility and the ease of use, it has not only become the most popular application download destination but also a hotbed for competing services to attract and gain customers. This project aims to employ machine learning & visual analytics concepts to gain insights into how applications become successful and achieve high user ratings.</p>

<p>The dataset chosen for this project was from __. It contains over 1M application data capturing various details like app name, description, genre, reviews, etc. The aim of the  project was the first to find an accurate machine learning model which could fairly accurately predict user ratings on any app when similar data is available. Subsequently, two different machine learning models were used and trained on this data.</p>

<p>Visualizations indicated that the apps were distributed across 48 distinct genre and that the  "Entertainment"  genre has the most user's category was the most popular within this dataset. It also showed that the user ratings in the dataset were mostly between 3.0 to 5.0. In the latter rating's interval the distribution roughly followed a normal distribution with the peak at approximate ratings of 4.5. Correlations among some major parameters were also visualized for the data. After initial visualizing and data processing, the goal was to create a machine learning model to predict user ratings. Two different models were created, and they were trained on the available data. The model (working on app name and description only) predicted user ratings with the least error rates and better when compared to the other one(training the model taken the app name, description, genre, and content rating). Finally, the important parameters responsible for predicting, it's the App name and description.</p>


<h4>Hypothesis<h4>
Predict Google Play Store app ratings.
  
<h4> Independent Variables</h4>
<ul>
<li>Title: Application name</li>
<li>Url: Application url</li>
<li>Reviews: Number of user reviews for the app (as when scraped)</li>
<li>Downloads: Number of user downloads for the app</li>
<li>Price: Paid or Free</li>
<li>Content Rating: Age group the app is targeted at - Children / Mature 21+ / Adult/ Teen</li>
<li>Genre: An app can belong to multiple genres (apart from its main category). For eg, a musical family game will belong to Music, Game, Family genres.</li>
</ul>

<h4>Dependent Variables (Target varible)</h4>
<li>Store_rating: Overall user rating of the app </li>
