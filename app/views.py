from flask import render_template, flash, redirect, url_for, jsonify, request, session, escape, abort
from app import app
from .forms import SearchForm
import os
import manager
from clustering import *

app.secret_key = 'azdazdzadza'
# INIT FREESOUND CLIENT API
c = manager.Client()

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return logout()

@app.route('/login', methods=['POST'])
def do_admin_login():  
    session['logged_in'] = True
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        flash('Search requested with query: %s' %form.query.data)
        return redirect(url_for('results', query=form.query.data))
    return render_template('search.html', 
                           title='Search',
                           form=form)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()

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

#@app.route('/')
#def index():
#    return render_template('index.html')


# When receiving query from ajax
@app.route('/_cluster')
def cluster():
    #c = manager.Client()
    query = request.args.get('query', None, type=str)
    res = c.my_text_search(query=query, fields="tags,analysis,description", descriptors="lowlevel.mfcc.mean")
    b = c.new_basket()
    b.load_sounds(res)
    #cluster = Cluster(basket=b)
    w2v = W2v(basket=b)
    cluster = w2v.run()
    cluster.run(k_nn=res.count/50)
    dict_list = []
    for k in range(len(cluster.tags_oc)):
        dict_list.append([{"text":cluster.tags_oc[k][i][0], "size":60.0*cluster.tags_oc[k][i][1]/max([cluster.tags_oc[k][i][1] for i in range(len(cluster.tags_oc[k]))])} for i in range(len(cluster.tags_oc[k]))])
    session['clusters'] = dict_list
    return jsonify(result=dict_list)

@app.route('/cluster')
def display():
    return render_template('clusters2.html')

@app.route('/_click')
def click():
    nb = request.args.get('cluster_num', None, type=int)
    print 'cluster: ' + str(nb)
    return jsonify(None)
    
@app.route('/tree')
def tree():
    return render_template('tree.html')