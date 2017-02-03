from flask import render_template, flash, redirect, url_for, jsonify, request
from app import app
from .forms import SearchForm
import manager
from clustering import *


# INIT FREESOUND CLIENT API
#c = manager.Client()


#@app.route("/")
#def hello():
#    return "Welcome to Python Flask App!"

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        flash('Search requested with query: %s' %form.query.data)
        return redirect(url_for('results', query=form.query.data))
    return render_template('search.html', 
                           title='Search',
                           form=form)

embed_blocks = ['https://www.freesound.org/embed/sound/iframe/', '/simple/medium/']
def create_embed(freesound_id):
    return embed_blocks[0] + str(freesound_id) + embed_blocks[1]

@app.route('/results/<query>')
def results(query):
    # TODO: need a function for iterating through pages
    # ADD PAGINATION TO MY INTERFACE
    # CONSIDER INDEXING METADATA AND ACOUSTIC FEATURES
    sounds = c.text_search(query=query, fields="id", page_size=20)
    results = [create_embed(s.id) for s in sounds]
    return render_template('results.html',
                           results=results)

# EX FROM WEB
@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/')
def index():
    return render_template('index.html')


# EX FROM WEB
@app.route('/_cluster')
def cluster():
    c = manager.Client()
    query = request.args.get('query', None, type=str)
    res = c.my_text_search(query=query, fields="tags,analysis", descriptors="lowlevel.mfcc.mean")
    b = c.new_basket()
    b.load_sounds(res)
    cluster = Cluster(basket=b)
    cluster.run()
    return jsonify(result=cluster.ids_in_clusters)

@app.route('/cluster')
def display():
    return render_template('clusters2.html')
