import freesound
# TODO:add imports
# create functions for geting metadata from db
# create db for metadata (or use the freesound one)
# create db for acoustic descritors


class Client(freesound.FreesoundClient):
    def my_text_search(self, **param):
        """
        Call text_search method from freesound.py and add all the defaults fields and page size parameters
        TODO : add default param more flexible (store in a param file - add the api_key in a .py file)

        >>> import manager
        >>> c = manager.Client()
        >>> result = c.my_text_search(query="wind")
        """
        
        fields = 'id,'
        try:
            fields += param['fields']
            param.pop('fields')
        except:
            pass
        results_pager = self.text_search(fields=fields, page_size=150, **param)
        #self.text_search(fields="id,name,url,tags,description,type,previews,filesize,bitrate,bitdepth,duration,samplerate,username,comments,num_comments,analysis_frames",page_size=150,**param)
        return results_pager

    def my_get_sound(self,idToLoad):
        """
        Use this method to get a sound from local or freesound if not in local
        >>> sound = c.my_get_sound(id)
        """
        # LOAD SOUND FROM DATABASE
        settings = SettingsSingleton()
        if idToLoad not in settings.local_sounds:
            sound = self._load_sound_freesound(idToLoad)
            if settings.autoSave:
                self._save_sound_json(sound)  # save it
        else:
            sound = self._load_sound_json(idToLoad)

        return sound
    


#_________________________________________________________________#
#                       Analysis class                            #
#_________________________________________________________________#
class Analysis():
    """
    Analysis nested object. Holds all the analysis of many sounds

    """
    def __init__(self, json_dict = None):
        if not json_dict:
            with open('analysis_template.json') as infile:
                json_dict = simplejson.load(infile)

        self.json_dict = json_dict
        def replace_dashes(d):
            for k, v in d.items():
                if "-" in k:
                    d[k.replace("-", "_")] = d[k]
                    del d[k]
                if isinstance(v, dict): replace_dashes(v)

        replace_dashes(json_dict)
        self.__dict__.update(json_dict)
        for k, v in json_dict.items():
            if isinstance(v, dict):
                self.__dict__[k] = Analysis(v)

    def rsetattr(self, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(pre) if pre else self, post, val)

    sentinel = object()
    def rgetattr(self, attr, default=sentinel):
        if default is self.sentinel:
            _getattr = getattr
        else:
            def _getattr(obj, name):
                return getattr(obj, name, default)
        return reduce(_getattr, [self] + attr.split('.'))

    def remove(self, index, descriptor):
        if index == 'all':
            self.rsetattr(descriptor, [])
        else:
            analysis = self.rgetattr(descriptor)
            del analysis[index]


#_________________________________________________________________#
#                        Basket class                             #
#_________________________________________________________________#
class Basket:
    """
    A basket where sounds and analysis can be loaded
    >>> c = manager.Client()
    >>> b = c.new_basket()
    TODO : add comments attribute, title...
    """

    def __init__(self):
        self.sounds = []
        self.analysis = Analysis() # the use of the nested object is not rly good...
        self.analysis_stats = []
        self.analysis_stats_names = []
        self.ids = []
        self.analysis_names = []

    def __add__(self, other):
        """
        Concatenate two baskets
        TODO : adapt it to new changes & make sure the order is not broken
        """
        sumBasket = copy.deepcopy(self)
        for i in range(len(other.sounds)):
            sumBasket.ids.append(other.ids[i])
            sumBasket.sounds.append(other.sounds[i])
        sumBasket._remove_duplicate()
        return sumBasket

    def __sub__(self, other):
        """
        Return a basket with elements of self that are not in other
        """
        subBasket = copy.deepcopy(self)
        idx_to_remove = [x[0] for x in enumerate(self.ids) if x[1] in other.ids]
        subBasket.remove(idx_to_remove)
        return subBasket
        
    def __len__(self):
        return len(self.ids)
         
    def _actualize(self): # used when an old basket is loaded from pickle
        if not hasattr(self, 'analysis_stats'):
            self.analysis_stats = []

    def _remove_duplicate(self):
        # TODO : add method to concatenate analysis in Analysis() (won't have to reload json...)
        ids_old = self.ids
        sounds_old = self.sounds
        self.ids = []
        self.sounds = []
        nbSounds = len(ids_old)
        for i in range(nbSounds):
            if ids_old[i] not in self.ids:
                self.ids.append(ids_old[i])
                self.sounds.append(sounds_old[i])
        self.update_analysis()
    
    #________________________________________________________________________#
    # __________________________ Users functions ____________________________#
    def push(self, sound, analysis_stat=None):
        """
        >>> sound = c.my_get_sound(query='wind')
        >>> b.push(sound)

        """
        #sound.name = strip_non_ascii(sound.name)
        self.sounds.append(sound)
        self.analysis_stats.append(analysis_stat)
        if sound is not None:
            self.ids.append(sound.id)      
        else:
            self.ids.append(None)

    def push_list_id(self, sounds_id):
        Bar = ProgressBar(len(sounds_id), LENGTH_BAR, 'Loading sounds')
        Bar.update(0)
        for idx, id in enumerate(sounds_id):
            sound = self.parent_client.my_get_sound(id)
            self.push(sound)
            Bar.update(idx+1)

    def remove(self, index_list):
        index_list = sorted(index_list, reverse=True)
        for i in index_list:
            del self.ids[i]
            del self.sounds[i]
            try:
                del self.analysis_stats[i]
            except IndexError:
                pass
            for descriptor in self.analysis_names:
                self.analysis.remove(i, descriptor)

    def remove_sounds_with_no_analysis(self):
        list_idx_to_remove = []
        for idx, analysis in enumerate(self.analysis_stats):
            if analysis is None:
                list_idx_to_remove.append(idx)
        self.remove(list_idx_to_remove)
                
    def update_sounds(self):
        """
        Use this method to load the sounds which ids are in the basket
        """
        nbSound = len(self.ids)
        Bar = ProgressBar(nbSound, LENGTH_BAR, 'Loading sounds')
        Bar.update(0)
        for i in range(nbSound):
            self.sounds.append(self.parent_client.my_get_sound(self.ids[i]))
            Bar.update(i+1)

    def add_analysis(self, descriptor):
        """
        Use this method to add the analysis.
        All the current loaded analysis will be erased
        All the analysis of the loaded sound ids will be loaded

        >>> results_pager = c.my_text_search(query='wind')
        >>> b.load_sounds(results_pager)
        >>> b.add_analysis('lowlevel.mfcc')
        """
        if descriptor in self.analysis_names:
            print 'The %s analysis are already loaded' % descriptor
        else:
            nbSound = len(self.ids)
            allFrames = []
            Bar = ProgressBar(nbSound,LENGTH_BAR, 'Loading ' + descriptor + ' analysis')
            Bar.update(0)
            for i in range(nbSound):
                allFrames.append(self.parent_client.my_get_analysis(self.ids[i], descriptor))
                Bar.update(i+1)
            self.analysis_names.append(descriptor)
            self.analysis.rsetattr(descriptor, allFrames)

    def update_analysis(self):
        for nameAnalysis in self.analysis_names:
            allFrames = self.analysis.rgetattr(nameAnalysis)
            nbAnalysis = len(allFrames)
            nbAnalysisToLoad = len(self.ids) - nbAnalysis
            Bar = ProgressBar(nbAnalysisToLoad, LENGTH_BAR, 'Loading ' + nameAnalysis + ' analysis')
            Bar.update(0)
            for i in range(nbAnalysisToLoad):
                Bar.update(i + 1)
                allFrames.append(self.parent_client.my_get_analysis(self.ids[i+nbAnalysis], nameAnalysis))

    def add_analysis_stats(self):
        """
        Use this method to add all analysis stats to all sounds in the basket
        (means and var of descriptors)
        """
        #self.analysis_stats = []
        nbSounds = len(self.sounds)
        Bar = ProgressBar(nbSounds, LENGTH_BAR, 'Loading analysis stats')
        Bar.update(0)
        for i, sound in enumerate(self.sounds):
            Bar.update(i + 1)
            if sound is not None:
                analysis = self.parent_client.my_get_analysis_stats(sound.id)
                self.analysis_stats[i] = analysis
            else:
                self.analysis_stats[i] = None # HERE CHANGED APPEND TO I, is it ok ?
                # try:
                #     self.analysis_stats.append(sound.get_analysis())
                # except freesound.FreesoundException:
                #     pass

	# FUNCTION FOR ADDING STATS OF ONLY ONE ANALYSIS
    def add_one_analysis_stats(self, descriptor):
        nbSounds = len(self.sounds)
        Bar = ProgressBar(nbSounds, LENGTH_BAR, 'Loading analysis stats')
        Bar.update(0)
        for i, sound in enumerate(self.sounds):
            Bar.update(i + 1)
            if sound is not None:
                analysis = self.parent_client.my_get_one_analysis_stats(sound.id, descriptor)
                self.analysis_stats[i] = analysis
            else:
                self.analysis_stats[i] = None
		
    def remove_analysis(self, descriptor):
        if descriptor in self.analysis_names:
            self.analysis.remove('all', descriptor)
            self.analysis_names.remove(descriptor)

    def load_sounds_(self, results_pager, begin_idx=0, debugger=None):
        """ 
        IN PROGRESS
        This function is used when the data to load in the basket is in the pager (and not just the id like for the next function)
        """
        nbSound = results_pager.count
        numSound = begin_idx # for iteration
        results_pager_last = results_pager
        Bar = ProgressBar(nbSound,LENGTH_BAR,'Loading sounds')
        Bar.update(0)
        # 1st iteration                              # maybe there is a better way to iterate through pages...
        for sound in results_pager:
            self.push(sound, sound.analysis)
            numSound = numSound+1
            Bar.update(numSound+1)

        # next iteration
        while (numSound<nbSound):
            count = 0
            while 1: # care with this infinite loop...
                count += 1
                if count>10: # MAYBE SOME BUG HERE
                    print 'could not get more sounds'
                    break
                try:
                    results_pager = results_pager_last.next_page()
                    if debugger:
                        debugger.append(results_pager)
                    break
                except:
                    exc_info = sys.exc_info()
                    sleep(1)
                    print exc_info
            for sound in results_pager:
                self.push(sound, sound.analysis)
                numSound = numSound+1
                Bar.update(numSound+1)
            results_pager_last = results_pager

    def extract_descriptor_stats(self, scale=False):
        """
        Returns a list of the scaled and concatenated descriptor stats - mean and var (all the one that are loaded in the Basket) for all sounds in the Basket.
        """
        feature_vector = []
        for analysis_stats in self.analysis_stats:
            feature_vector_single_sound = []
            for k, v in analysis_stats.as_dict().iteritems():
                if k == 'lowlevel':
                    for k_, v_ in v.iteritems():
                        try: # some lowlevel descriptors do not have 'mean' 'var' field (eg average_loudness)    
                            # barkbands_kurtosis has 0 variance and that bring dvar and dvar2 to be None...
                            if isinstance(v_['mean'], list):
                                feature_vector_single_sound += v_['mean'] # take the mean
                                feature_vector_single_sound += v_['dmean']
                                feature_vector_single_sound += v_['dmean2']
                                feature_vector_single_sound += v_['var'] # var
                                feature_vector_single_sound += v_['dvar']
                                feature_vector_single_sound += v_['dvar2']                                
                            elif isinstance(v_['mean'], float):
                                feature_vector_single_sound.append(v_['mean']) # for non array
                                feature_vector_single_sound.append(v_['dmean'])
                                feature_vector_single_sound.append(v_['dmean2'])
                                feature_vector_single_sound.append(v_['var'])
                                if k_ != 'barkbands_kurtosis': # this descriptor has variance = 0 => produce None values for dvar and dvar2
                                    feature_vector_single_sound.append(v_['dvar'])
                                    feature_vector_single_sound.append(v_['dvar2'])
                        except: # here we suppose that v_ is already a number to be stored 
                            if isinstance(v_, list):
                                feature_vector_single_sound += v_
                            elif isinstance(v_, float):
                                feature_vector_single_sound.append(v_)
                elif k == 'other cat of descriptors':
                    # sfx, tonal, rhythm
                    pass
            feature_vector.append(feature_vector_single_sound)
        if scale:  
            return preprocessing.scale(feature_vector)
        else:
            return feature_vector
        
    def extract_one_descriptor_stats(self, scale=False):
        """
        A bit dirty. Maybe review de concept of analysis_stat and analysis objects
        """
        feature_vector = []
        for analysis_stats in self.analysis_stats:
            feature_vector_single_sound = []
            if isinstance(getattr(analysis_stats,'mean'), list):
                feature_vector_single_sound += getattr(analysis_stats,'mean') # take the mean
                feature_vector_single_sound += getattr(analysis_stats,'dmean')
                feature_vector_single_sound += getattr(analysis_stats,'dmean2')
                feature_vector_single_sound += getattr(analysis_stats,'var') # var
                feature_vector_single_sound += getattr(analysis_stats,'dvar')
                feature_vector_single_sound += getattr(analysis_stats,'dvar2')                                
            elif isinstance(getattr(analysis_stats,'mean'), float):
                feature_vector_single_sound.append(getattr(analysis_stats,'mean')) # for non array
                feature_vector_single_sound.append(getattr(analysis_stats,'dmean'))
                feature_vector_single_sound.append(getattr(analysis_stats,'dmean2'))
                feature_vector_single_sound.append(getattr(analysis_stats,'var'))
                if k_ != 'barkbands_kurtosis': # this descriptor has variance = 0 => produce None values for dvar and dvar2
                    feature_vector_single_sound.append(getattr(analysis_stats,'dvar'))
                    feature_vector_single_sound.append(getattr(analysis_stats,'dvar2'))
            feature_vector.append(feature_vector_single_sound)
        if scale:  
            return preprocessing.scale(feature_vector)
        else:
            return feature_vector
        
    
    def load_sounds(self, results_pager, begin_idx=0, debugger=None):
        """
        Use this method to load all the sounds from a result pager int the basket
        this method does not take the objects from the pager but usgin my_get_sound() which return a sound with all the fields

        >>> results_pager = c.my_text_search(query='wind')
        >>> b.load_sounds(results_pager)
        """
        nbSound = results_pager.count
        numSound = begin_idx # for iteration
        results_pager_last = results_pager
        Bar = ProgressBar(nbSound,LENGTH_BAR,'Loading sounds')
        Bar.update(0)
        # 1st iteration                              # maybe there is a better way to iterate through pages...
        for i in results_pager:
            self.push(self.parent_client.my_get_sound(i.id),analysis_stat=None)
            numSound = numSound+1
            Bar.update(numSound+1)

        # next iteration
        while (numSound<nbSound):
            count = 0
            while 1: # care with this infinite loop...
                count += 1
                if count>10: # MAYBE SOME BUG HERE
                    print 'could not get more sounds'
                    break
                try:
                    results_pager = results_pager_last.next_page()
                    if debugger:
                        debugger.append(results_pager)
                    break
                except:
                    exc_info = sys.exc_info()
                    sleep(1)
                    print exc_info
            for i in results_pager:
                self.push(self.parent_client.my_get_sound(i.id),analysis_stat=None)
                numSound = numSound+1
                Bar.update(numSound+1)
            results_pager_last = results_pager

    def retrieve_previews(self, new_folder = None):
        folder = './previews/'
        if new_folder is not None:
            folder += new_folder
            if not os.path.exists(folder):
                os.makedirs(folder) 
        nbSounds = len(self.sounds)
        Bar = ProgressBar(nbSounds, LENGTH_BAR, 'Downloading previews')
        Bar.update(0)
        for i in range(nbSounds):
            Bar.update(i+1)
            self.sounds[i].retrieve_preview(folder)

    def save(self, name):
        """
        Use this method to save a basket
        Only ids and analysis name(s) are saved in a list [ [id1,...idn], [analysis, ...] ]
        TODO : change it and save it as a dict (more flexible and stable regarding changes)
        """
        settings = SettingsSingleton()
        if name and not (name in settings.local_baskets):
            basket = [self.ids]
            basket.append(self.analysis_names)
            nameFile = 'baskets/' + name + '.json'
            with open(nameFile, 'w') as outfile:
                json.dump(basket, outfile)
            settings.local_baskets.append(name)
        else:
            overwrite = raw_input(name + ' basket already exists. Do you want to replace it ? (y/n)')
            if overwrite == 'y':
                settings.local_baskets.remove(name)
                self.save(name)
            else:
                print 'Basket was not saved'

    def load(self,name):
        """
        Use thise method to load a basket from json files
        """
        self.sounds = []
        settings = SettingsSingleton()
        if name and name in settings.local_baskets:
            nameFile = 'baskets/' + name + '.json'
            with open(nameFile) as infile:
                basket = simplejson.load(infile)
            ids = basket[0]
            nbSounds = len(ids)
            for i in range(nbSounds):
                self.ids.append(ids[i])
            self.update_sounds()
            self.analysis_names = basket[1]
            self.update_analysis()
        else:
            print '%s basket does not exist' % name

    def save_pickle(self, name):
        settings = SettingsSingleton()
        if name and not (name in settings.local_baskets_pickle):
            self.parent_client.save_pickle(self, name, 'baskets_pickle/')
            settings.local_baskets_pickle.append(name)
        else:
            overwrite = raw_input(name + ' basket already exists. Do you want to replace it ? (y/n)')
            if overwrite == 'y':
                self.parent_client.save_pickle(self, name, 'baskets_pickle/')
            else:
                print 'Basket was not saved'

    #________________________________________________________________________#
    # __________________________ Language tools _____________________________#
    # TODO: CREATE A CLASS FOR THIS TOOLS, AND SEPARATE FROM BASKET 
    
    def tags_occurrences(self):
        """
        Returns a list of tuples (tag, nb_occurrences, [sound ids])
        The list is sorted by number of occurrences of tags
        """
        all_tags_occurrences = []
        tags = self.tags_extract_all()
        Bar = ProgressBar(len(tags), LENGTH_BAR, 'Thinking ...')
        Bar.update(0)
        for idx, tag in enumerate(tags):
            Bar.update(idx+1)
            tag_occurrences = self.tag_occurrences(tag)
            all_tags_occurrences.append((tag, tag_occurrences[0], tag_occurrences[1]))
        all_tags_occurrences = sorted(all_tags_occurrences, key=lambda oc: oc[1])
        all_tags_occurrences.reverse()
        return all_tags_occurrences

    def terms_occurrences(self, terms_sounds):
        """
        Input: list of list of terms for each sound
        Returns a list of tuples (terms, nb_occurrences, [sound ids])
        The list is sorted by number of occurrences of tags
        Typicaly:   t = basket.preprocessing_tag_description()
                    t_o = basket.terms_occurrences(t)
                    nlp(basket, t_o) 
                    WARNING: nlp check the tags only... !!!!!!!!!!
        """
        all_terms_occurrences = []
        terms = list(set([item for sublist in terms_sounds for item in sublist]))
        Bar = ProgressBar(len(terms), LENGTH_BAR, 'Thinking ...')
        Bar.update(0)
        for idx, term in enumerate(terms):
            Bar.update(idx+1)
            term_occurrences = self.term_occurrences(terms_sounds, term)
            all_terms_occurrences.append((term, term_occurrences[0], term_occurrences[1]))
        all_terms_occurrences = sorted(all_terms_occurrences, key=lambda oc: oc[1])
        all_terms_occurrences.reverse()
        return all_terms_occurrences

    def term_occurrences(self, l, term):
        ids = []
        for i, sound_terms in enumerate(l):
            if term in sound_terms:
                ids.append(i)
        number = len(ids)
        return number, ids
        
    def tag_occurrences(self, tag):
        ids = []
        for i, sound in enumerate(self.sounds):
            if sound is not None:
                if tag in sound.tags:
                    ids.append(i)
            number = len(ids)
        return number, ids

    def description_occurrences(self, stri):
        ids = []
        for i in range(len(self.sounds)):
            if stri in self.sounds[i].description:
                ids.append(i)
        number = len(ids)
        return number, ids

    def tags_extract_all(self):
        tags = []
        Bar = ProgressBar(len(self.sounds), LENGTH_BAR, 'Extracting tags')
        Bar.update(0)
        for idx, sound in enumerate(self.sounds):
            Bar.update(idx + 1)
            if sound is not None:
                for tag in sound.tags:
                    if tag not in tags:
                        tags.append(tag)
        return tags
    
    def create_sound_tag_dict(self):
        """
        Returns a dictionary with sound id in keys and tags in values
        """
        sound_tag_dict = {}
        for sound in self.sounds:
            sound_tag_dict[sound.id] = sound.tags
        return sound_tag_dict
    
    def get_preprocessed_descriptions_word2vec(self):
        """
        Returns a list of sentences from sound descriptions in the basket.
        Preprocessing is done (remove special characters, Porter Stemming, lower case)
        """
        stemmer = PorterStemmer()
        delimiters = '.', '?', '!', ':'
        def split(delimiters, string, maxsplit=0):
            regexPattern = '|'.join(map(re.escape, delimiters))
            return re.split(regexPattern, string, maxsplit)
        
        all_descriptions = [a.description.lower() for a in self.sounds]
        sentences = []
        
        for description in all_descriptions:
            string = description.replace('\r\n', ' ')
            string = string.replace('(', ' ')
            string = string.replace(')', ' ')
            string = string.replace('*', '')
            string = string.replace('-', '')
            string = string.replace('#', '')
            string = string.replace(',', '')
            string = string.replace('/', '')
            string = re.sub('<a href(.)+>', ' ', string)
            string = split(delimiters, string)
            for string_sentence in string:
                if string_sentence is not u'':
                    terms_to_append = [stemmer.stem(a) for a in string_sentence.split()]
                    sentences.append(terms_to_append)
    
        return sentences
    
    def word2vec(self, sentences, size=50):
        from gensim.models import Word2Vec
        return Word2Vec(sentences, size=size, window=500, min_count=10, workers=8)
    
    def doc2vec(self, documents, size=50):
        """ 
        This method seems to give worse result on returning most similar terms for violin, bright
        """
        from gensim.models import Doc2Vec
        return Doc2Vec(documents, size=size, window=500, min_count=10, workers=8)
    
    def preprocessing_tag_description(self):
        """
        Preprocessing tags and descriptions
        Returns an array containing arrays of terms for each sound
        Steps for descriptions : Lower case, remove urls, Tokenization, remove stop words, Stemming (Porter)
                    tags       : Lower case, Stemming
        """
        stemmer = PorterStemmer()
        en_stop = get_stop_words('en') + ['freesound', 'org']
        
        all_descriptions = [[stemmer.stem(word) for word in CountVectorizer().build_tokenizer()(re.sub('<a href(.)+/a>', ' ', sound.description.lower())) if word not in en_stop] for sound in self.sounds]
        all_tags = [[stemmer.stem(tag.lower()) for tag in sound.tags] for sound in self.sounds]
        
        return [tag + description for tag, description in zip(all_tags, all_descriptions)]
    
    def preprocessing_tag(self):
        stemmer = PorterStemmer()
        return [[stemmer.stem(tag.lower()) for tag in sound.tags] for sound in self.sounds]
    
    def preprocessing_doc2vec(self):
        from gensim.models.doc2vec import TaggedDocument
        stemmer = PorterStemmer()
        en_stop = get_stop_words('en') + ['freesound', 'org']
        
        all_descriptions = [[stemmer.stem(word) for word in CountVectorizer().build_tokenizer()(re.sub('<a href(.)+/a>', ' ', sound.description.lower())) if word not in en_stop] for sound in self.sounds]
        all_tags = [[stemmer.stem(tag.lower()) for tag in sound.tags] for sound in self.sounds]
        
        return [TaggedDocument(words, tags) for words, tags in zip(all_descriptions, all_tags)]
        
    class TfidfEmbeddingVectorizer(object):
        def __init__(self, w2v_model):
            self.word2vec = dict(zip(w2v_model.index2word, w2v_model.syn0))
            self.word2weight = None
            self.dim = len(w2v_model.syn0[0])

        def fit(self, X, y):
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(X)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of 
            # known idf's
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(
                lambda: max_idf,
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

            return self

        def transform(self, X):
            return np.array([
                    np.mean([self.word2vec[w] * self.word2weight[w]
                             for w in words if w in self.word2vec] or
                            [np.zeros(self.dim)], axis=0)
                    for words in X
                ])

#_________________________________________________________________#
#                           NLP class                             #
#_________________________________________________________________#
class Nlp:
    """ 
    Methods for creating sparse occurrences matrix, similarity, graphs, etc...
    """
    def __init__(self, basket, tags_occurrences = None):
        if tags_occurrences:
            self.tags_occurrences = tags_occurrences
        else:
            self.tags_occurrences = basket.tags_occurrences()
        self.set_tags = [tag[0] for tag in self.tags_occurrences]
        self.freesound_sound_id = [sound.id for sound in basket.sounds]
        self.sound_tags = [sound.tags for sound in basket.sounds]
        self.inverted_tag_index = self._inverted_tag_index(self.set_tags)
        self.nb_sound = len(self.freesound_sound_id)
        self.nb_tag = len(self.set_tags)
    
    def _inverted_tag_index(self, set_tags):
        inverted_tag_index = dict()
        for idx, tag in enumerate(set_tags):
            inverted_tag_index[tag] = idx
        return inverted_tag_index   
    
    def create_sound_tag_matrix(self):
        """
        Returns scipy sparse matrix sound id / tag (2d array) - lil_matrix 
        Sounds are ordered like in the Basket (=self object)
        Tags are ordered like in the tags_occurrences list
        """
        Bar = ProgressBar(self.nb_sound, LENGTH_BAR, 'Creating matrix...')
        Bar.update(0)
        self.sound_tag_matrix = scipy.sparse.lil_matrix((self.nb_sound,self.nb_tag), dtype=int)
        for idx_sound, tags in enumerate(self.sound_tags):
            Bar.update(idx_sound+1)
            for tag in tags:
                self.sound_tag_matrix[idx_sound, self.inverted_tag_index[tag]] = 1
    
    def return_tag_cooccurrences_matrix(self):
        """
        Returns the tag to tag cooccurrences matrix by doing A_transpose * A where A is the sound to tag matrix occurrences
        """
        try:
            return self.sound_tag_matrix.transpose() * self.sound_tag_matrix      
        except:
            print 'Create fist the sound tag matrix using create_sound_tag_matrix method'
            
    @staticmethod
    def return_similarity_matrix_tags(tag_something_matrix):
        """
        Returns a tag similarity matrix computed with cosine distance from the given matrix
        MemoryError problem
        """
        tag_similarity_matrix = cosine_similarity(tag_something_matrix)
        return tag_similarity_matrix
    
    def return_my_similarity_matrix_tags(self, tag_something_matrix):
        """
        TOO SLOW !!!
        Returns a tag similarity matrix computed with cosine distance from the given matrix
        """
        size_matrix = tag_something_matrix.shape[0]
        tag_similarity_matrix = np.zeros(shape=(size_matrix,size_matrix), dtype='float32')
        Bar = ProgressBar(size_matrix*size_matrix/2, LENGTH_BAR, 'Calculating similarities...')
        Bar.update(0)
        for i0 in range(size_matrix):
            row0 = tag_something_matrix.getrow(i0).toarray()
            for i1 in range(i0):
                Bar.update(i0*size_matrix + i1 + 1)
                row1 = tag_something_matrix.getrow(i1).toarray()
                tag_similarity_matrix[i0][i1] = 1 - spatial.distance.cosine(row0, row1)
                tag_similarity_matrix[i1][i0] = tag_similarity_matrix[i0][i1]
        return tag_similarity_matrix
    
    """
    PRINT SOME SIMILARITIES BTW TAGS:
    for i in range(200):
        print str(i).ljust(10) + set_tags[i].ljust(30) + str(sim[67,i])
    """
     
    def create_tag_sound_matrix(self, tags_occurrences):
        """
        DO NOT USE THIS - TODO: implement it like sound_tag_matrix. Or just call create_sound_tag_matrix and transpose it...
        Returns a matrix tag / sound id 
        Ordered like in tags_occurrences and in the Basket (=self)
        """
        tag_sound_matrix = []
        for tag in tags_occurrences:
            sound_vect = [0] * len(self.nb_sounds)
            for sound_id_in_basket in tag[2]:
                sound_vect[sound_id_in_basket] = 1
            tag_sound_matrix.append(sound_vect)
        return tag_sound_matrix    
    
    @staticmethod
    def nearest_neighbors(similarity_matrix, idx, k):
        distances = []
        for x in range(len(similarity_matrix)):
            distances.append((x,similarity_matrix[idx][x]))
        distances.sort(key=operator.itemgetter(1), reverse=True)
        return [d[0] for d in distances[0:k]]
    
    # __________________ GRAPH __________________ #
    def create_knn_graph_igraph(self, similarity_matrix, k):
        """ Returns a knn graph from a similarity matrix - igraph module """
        np.fill_diagonal(similarity_matrix, 0) # for removing the 1 from diagonal
        g = ig.Graph(directed=True)
        g.add_vertices(len(similarity_matrix))
        g.vs["b_id"] = range(len(similarity_matrix))
        for idx in range(len(similarity_matrix)):
            g.add_edges([(idx, i) for i in self.nearest_neighbors(similarity_matrix, idx, k)])
            print idx, self.nearest_neighbors(similarity_matrix, idx, k)
        return g
    
    def create_knn_graph(self, similarity_matrix, k):
        """ Returns a knn graph from a similarity matrix - NetworkX module """
        np.fill_diagonal(similarity_matrix, 0) # for removing the 1 from diagonal
        g = nx.Graph()
        g.add_nodes_from(range(len(similarity_matrix)))
        for idx in range(len(similarity_matrix)):
            g.add_edges_from([(idx, i) for i in self.nearest_neighbors(similarity_matrix, idx, k)])
            print idx, self.nearest_neighbors(similarity_matrix, idx, k)
        return g
    
    # OLD
    def create_tag_similarity_graph(self, tag_similarity_matrix, tag_names, threshold):
        """
        TODO : ADAPT IT FOR NetworkX package
        Returns the tag similarity graph (unweighted) from the tag similarity matrix
        """
        g = Graph()
        g.add_vertices(len(tag_names))
        g.vs["name"] = tag_names
        g.vs["label"] = g.vs["name"]
        for tag_i in range(len(tag_similarity_matrix)):
            for tag_j in range(len(tag_similarity_matrix)):
                if tag_i < tag_j:
                    if tag_similarity_matrix[tag_i][tag_j] > threshold:
                        g.add_edge(tag_i, tag_j)
        return g
    
    def get_centrality_from_graph(self, graph):
        return g.evcent()
    
    # TODO : ORDER TAG BY CENTRALITY
    # name_cent = [ (t[i], cent[i]) for i in range(len(t))]
    # name_cent.sort(key=lambda x: x[1], reverse=True)

    # TODO : CREATE FUNCTION FOR CREATION OF TAXONOMY
#    g2 = Graph.Tree(2,1)
#    g2.add_vertices(58978)
#    g2.vs["name"] = names
#    g2.vs["label"] = g2.vs["name"]
#    list_tags_in_tax = [0]
#
#    for idx in range(58978):
#            idx = idx + 1
#            maxCandidateVal = 0
#            for tag_1 in list_tags_in_tax:
#                    if not tag_1 == idx:
#                            if s_m_t[idx][tag_1] > maxCandidateVal:
#                                    maxCandidateVal = s_m_t[idx][tag_1]
#                                    maxCandidate = tag_1
#            if maxCandidateVal > 0.5:
#                    g2.add_edge(tag_1+1,idx+1)
#                    print 'added edge'
#                    print maxCandidateVal
#            else:
#                    g2.add_edge(0,idx+1)
#                    print 'added edge to root'
#            list_tags_in_tax.append(idx)
    
    # TODO : PUT THIS GRAPH THINGS IN AN OTHER CLASS
    
    