from flask import render_template, flash, redirect, url_for
from app import app
from .forms import SearchForm
import freesound
from api_key import token

# INIT FREESOUND CLIENT API
c = freesound.FreesoundClient()
c.set_token(token)

@app.route("/")
def hello():
    return "Welcome to Python Flask App!"

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