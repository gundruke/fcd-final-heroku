import numpy as np
import tensorflow_text as text
import tensorflow as tf
from utils import pos_tagger, sentiment_analyzer

from flask import Flask, request, render_template,jsonify # Import flask libraries

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")
translator = tf.saved_model.load('translator')
pos_tags = ["ADJ", "NOUN", "PROPN", "VERB", "ADV"]

# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('home.html') # Render home.html

# Route 'classify' accepts GET request
@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        original_text = request.args.get("original_text") # Get original text

        # Get the output from the classification model
        translated_text = translator(original_text).numpy().decode()
        pt_tagger = pos_tagger(original_text, "PT", pos_tags)
        
        en_tagger = pos_tagger(translated_text, "EN", pos_tags)
        
        pt_sentiment, pt_polarity_dict = sentiment_analyzer(original_text, "PT")
        en_sentiment, en_polarity_dict = sentiment_analyzer(translated_text, "EN")



        # Render the output in new HTML page
        return render_template('output.html', 
                               original_text=original_text,
                               translated_text=translated_text,
                               pt_tagger=pt_tagger,
                               en_tagger=en_tagger, 
                               pt_sentiment = pt_sentiment,
                               pt_polarity_dict = pt_polarity_dict, 
                               en_sentiment = en_sentiment, 
                               en_polarity_dict = en_polarity_dict )
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        