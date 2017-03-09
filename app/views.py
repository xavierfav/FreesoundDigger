from flask import render_template, flash, redirect, url_for, jsonify, request, session, escape, abort
from app import app
from flask_session import Session
from .forms import SearchForm
import os
import manager
from clustering import *


app.secret_key = 'azdazdzwefwefadza'
# INIT FREESOUND CLIENT API
c = manager.Client()

SESSION_TYPE = 'redis'
app.config.from_object(__name__)
Session(app)


@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return logout()

@app.route('/login', methods=['POST'])
def do_admin_login():  
    session['logged_in'] = True
    #return render_template('index.html')
    return render_template('paginator.html')

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
    res = c.my_text_search(query=query, fields='tags')
    b = c.new_basket()
    b.load_sounds_(res)
    cluster = Cluster(basket=b)
    #w2v = W2v(basket=b)
    #cluster = w2v.run()
    cluster.run(k_nn=res.count/20, feature='fusion')
    #previews_list = [[s.previews.preview_lq_ogg for s in basket.sounds] for basket in cluster.cluster_baskets]
    #session['previews'] = previews_list[0]
    session['ids'] = [[s.id for s in basket.sounds] for basket in cluster.cluster_baskets]
    ids_in_clusters = [[s.id for s in cluster.cluster_baskets[basket_id].sounds[:5]] for basket_id in range(len(cluster.cluster_baskets))]
    return jsonify(result=ids_in_clusters)

@app.route('/cluster')
def display():
    return render_template('clusters2.html')

@app.route('/_click')
def click():
    nb = request.args.get('cluster_num', None, type=int)
    print 'cluster: ' + str(nb)
    print session.get('previews')
    return jsonify(result=session.get('previews'))
    
@app.route('/tree')
def tree():
    return render_template('tree.html')

@app.route('/paginator')
def page():
    return render_template('paginator.html')

@app.route('/_query')
def query_cluster():
    query = request.args.get('query', None, type=str)
    res = c.my_text_search(query=query, fields='tags')
    b = c.new_basket()
    b.load_sounds_(res)
    cluster = Cluster(basket=b)
    #w2v = W2v(basket=b)
    #cluster = w2v.run()
    cluster.run(k_nn=res.count/10, feature='fusion')
    session['ids'] = dict([(basket_id, [s.id for s in cluster.cluster_basket[basket_id].sounds]) for basket_id in range(len(cluster.cluster_basket))])
    ids_in_clusters = dict([[s.id for s in cluster.cluster_basket[basket_id].sounds[:5]] for basket_id in range(2)])#   range(len(cluster.cluster_basket))])
    return jsonify(result=ids_in_clusters)

@app.route('/_get_sound_id')
def send_sound_ids():
    page = request.args.get('page', None, type=int)
    cluster_id = request.args.get('cluster_id', None, type=int)
    #print page, cluster_id
    ids_in_clusters = session.get('ids')
    ids = [idx for idx in ids_in_clusters[cluster_id][page*5:page*5+5]]
    return jsonify(list_ids=ids)