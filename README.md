# Predict Google Play Store App Ratings
<h3>Training Neural Network for Classification</h3>

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
nltk.download('stopwords') 

<h5>Introduction</h5>

<p>The ability to use services and products on the go has been a major leap in this century. Applications on the Google play store aim to do exactly that. Owing to worldwide accessibility and the ease of use, it has not only become the most popular application download destination but also a hotbed for competing services to attract and gain customers. This project aims to employ machine learning & visual analytics concepts to gain insights into how applications become successful and achieve high user ratings.</p>

<p>The dataset chosen for this project was from __. It contains over 1M application data capturing various details like app name, description, genre, reviews, etc. The aim of the  project was the first to find an accurate machine learning model which could fairly accurately predict user ratings on any app when similar data is available. Subsequently, two different machine learning models were used and trained on this data.</p>

<p>Visualizations indicated that the apps were distributed across 48 distinct genre and that the  "Entertainment"  genre has the most user's category was the most popular within this dataset. It also showed that the user ratings in the dataset were mostly between 3.0 to 5.0. In the latter rating's interval the distribution roughly followed a normal distribution with the peak at approximate ratings of 4.5. Correlations among some major parameters were also visualized for the data. After initial visualizing and data processing, the goal was to create a machine learning model to predict user ratings. Two different models were created, and they were trained on the available data. The model (working on app name and description only) predicted user ratings with the least error rates and better when compared to the other one(training the model taken the app name, description, genre, and content rating). Finally, the important parameters responsible for predicting, it's the App name and description.</p>


<h4>Hypothesis</h4>
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

<h4>Dependent Variables (Target Varible)</h4>
  <li>Store_rating: Overall user rating of the app </li>
  
  
<h4>Data Visualization</h4>
<p>We visualize at the beginning the distribution of the “apps ratings”. We notice that “the fit” of the dataset rating does not follow -technically- a normal distribution with a mean of 4.08
In my opinion, Before the user's downloading the app first we see the rating of the app if the app rating is more than 4 then we say that yeah this is a good category of the app then after we see another attribute of apps. Very few people see the reviews of the app.
That's why the data distribution look's biased to apps that have a rating of more than 3.5</p>

![rating](https://user-images.githubusercontent.com/47077167/102839396-186b1f00-4409-11eb-9e79-6e7feba85346.jpeg)
You can open Data_Exploration file for more visualization about the data  

<h3>Data Preprocessing</h3>
<p>Preprocessing is important into transitioning raw data into a more desirable format. Undergoing the preprocessing process can help to get more accuracy in the prediction</p>
<p>In this section we will look at basic pre-processing of text data that must be done in order to make our data ready to be used as an input into our LSTM model</p>

<h6>Here, the first step was to merge the app name and description into one field called text (becuse we need a sequence of text for lstm model)</h6>
<h6> Then we cleaning the text, using multiple methods </h6>
<ul>
<li>Cleaning the text through remove the numbers, symbols and convert to lower case </li>
<li>Remove the links in the text,including the link that does not have a protocol like facebook.com</li>
<li>Remove the stop words from the text e.g. the,of,on,with, etc..</li>
<li>stemming the word by reducing the word to its core root e.g. the words ending with “ed”, or “ing”</li>
<li> Lemmatization is closely related to stemming. 
    It goes a steps further by linking words with similar meaning to one word.
    e.g. better -> good / was -> be </li>
  <li> Remove the words that contains numbers or symbols</li>

</ul>



<h3>Prediction models</h3>
<p>RNNs are a family of neural networks that are suitable for learning sequential data. Since our data consists of apps names and descriptions, we know for sure that this is a type of sequential data. 


In a recurrent neural network, we store the output activations from one or more of the layers of the network. Often these are hidden later activations. Then, the next time we feed an input example to the network, we include the previously-stored outputs as additional inputs.

RNNs have been made to address this specific issue. LSTM is just a variant of RNN which has been observed to give better results in comparison to vanilla RNNs in a variety of problems.

We use LSTM model to train the data 

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.And will help us to predict the expected rating for the app 

 </p>


<h4>We have two models(two cases):</h4>
1- In the first case we use the LSTM model to trainig the data 
<p>We train the model using the app name and description only </p>
<p> The first step we do, Cleaning the text. Back to the Data Preprocessing section</p>
<p> Text data has to be integer encoded before feeding it into the LSTM model. This can be easily achieved by using basic tools from the Keras library with only a few lines of code. Ttext should be tokenized by fitting Tokenizer class on the data set. As you can see I use “lower = True” argument to convert the text into lowercase to ensure consistency of the data. Afterwards, we should map our list of words (tokens) to a list of unique integers for each unique word using texts_to_sequences class.</p> 
<p>As an example, below you can see how the original reviews turn into a sequence of integers after applying prepocessing, tokenize and 
texts_to_sequences.
  
![tokenizer](https://user-images.githubusercontent.com/47077167/102868805-54bb7100-4443-11eb-911c-c02edd8e8789.jpeg)


<p>Next, we use pad_sequences class on the list of integers to ensure that all reviews have the same length, which is a very important step for preparing data for RNN model. Applying this class would either shorten the reviews to 100 integers, or pad them with 0’s in case they are shorter.</p>

![text_to_feaure](https://user-images.githubusercontent.com/47077167/102868602-0b6b2180-4443-11eb-8cda-61eced7fa32a.jpeg)


<p> It is time to build the model and fit it on the training data using Keras:
  As embedding requires the size of the vocabulary and the length of input sequences, we set vocabulary_size (MAX_SEQUENCE_LENGTH)equal to 500 and input_length at 50000 (max_words). Embedding size parameter specifies how many dimensions will be used to represent each word we take 100 as an input for this parameter(EMBEDDING_DIM)<p>
  
  <p>Next, we add 1 hidden LSTM layer with 100 memory cells. Potentially, adding more layers and cells can lead to better results.</p>
  <p>Finally, we add the output layer with softmax activation function </p>
  
  <p>After training the model for 3 epochs, we achieve an accuracy of 0.5362</p>
  
  <h6> Summry</h6>
  This model built to predict rating (based on attributes: app name, description, genre, and content rating) have accuracy on the test dataset 0.5362
The accuracy not perfect due to the distribution of the data we have is skewed. As we notice before, most of the apps rating range between 4 - 5 and 1,2,3 rating have a few samples.

---------------------------------------------------------------------------------------

In the second model, we built the model based on app name and description attributes

To increase the accuracy we need more samples that have 1,2,3 rating. So we merge the samples that have 1 and 2 rating to one class name 1 and the samples have 3 rating to 2 as a new class and merge 4,5 rating to 3 as a new class


-----------------------------------------------------------------------------------------

<h4>Future plan to improve the accuracy value on testing and training </h4>


<ol>
<li>Build a layer for genre and a layer for content rating instead of merge it with the text sequence for (description and app name)
</li>
<li>Then we need to exploit the review's feature to merge it with the text sequence, this called  multiple input models.
We will use a bidirectional LSTM model and combine its output (text)with the metadata(numerical). Therefore we define two input layers and treat them in separate models (nlp_input and meta_input). Our NLP data goes through the embedding transformation and the LSTM layer. The meta data is just used as it is, so we can just concatenate it with the lstm output (nlp_out). This combined vector is now classified in a dense layer and finally sigmoid in to the output neuron.
</li>
</ol>

