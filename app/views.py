from flask import render_template, flash, redirect
from app import app
from .forms import SearchForm


@app.route("/")
def hello():
    return "Welcome to Python Flask App!"

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        flash('Search requested with query: %s' %form.query.data)
        return redirect('/results')
    return render_template('search.html', 
                           title='Search',
                           form=form)

embed_blocks = ['https://www.freesound.org/embed/sound/iframe/', '/simple/medium/']
def create_embed(freesound_id):
    return embed_blocks[0] + str(freesound_id) + embed_blocks[1]

@app.route('/results')
def results():
    results = []
    results.append(create_embed(86807))
    return render_template('results.html',
                           results=results)