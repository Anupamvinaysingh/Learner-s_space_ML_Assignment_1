There are some differences in the three methods:-
1. The output of CountVectorizer method is higher then the rest because for calculating tf, it count the absolute frequency of each word but in manual method, we take normalised frequency of words(frequency/total words).
2. The output for words repeated in all the lines(e.g. the) is 0 for manual and CountVectorizer method but not for tfidfmethod, also its value is different from manual because in tfidfvectorizer, the system automatically uses the formula idf=log((1+N)/(1+df)) + 1, but in manual and countvectoriser, we use idf=log((N)/(df))
3. The one letter words (a) are ignored in CountVectorizer and tfidfvectorizer.

Comparison of outputs:-
Method 1:- Manual TF- IDF
corpus 1 TF-IDF:
star: 0.2197
sun: 0.0811
is: 0.0811
a: 0.0811
the: 0.0000

corpus 2 TF-IDF:
satellite: 0.2197
moon: 0.0811
is: 0.0811
a: 0.0811
the: 0.0000

corpus 3 TF-IDF:
and: 0.1569
are: 0.1569
celestial: 0.1569
bodies: 0.1569
sun: 0.0579
moon: 0.0579
the: 0.0000

Method 2:- CountVectorizer
Corpus 1 TF-IDF:
star: 1.0986
is: 0.4055
sun: 0.4055
the: 0.0000

Corpus 2 TF-IDF:
satellite: 1.0986
is: 0.4055
moon: 0.4055
the: 0.0000

Corpus 3 TF-IDF:
and: 1.0986
are: 1.0986
bodies: 1.0986
celestial: 1.0986
moon: 0.4055
sun: 0.4055
the: 0.0000

Method- 3 TfidfVectorizer
Corpus 1 TF-IDF:
star: 0.6317
is: 0.4805
sun: 0.4805
the: 0.3731

Corpus 2 TF-IDF:
satellite: 0.6317
is: 0.4805
moon: 0.4805
the: 0.3731

Corpus 3 TF-IDF:
and: 0.4262
are: 0.4262
bodies: 0.4262
celestial: 0.4262
moon: 0.3241
sun: 0.3241
the: 0.2517
