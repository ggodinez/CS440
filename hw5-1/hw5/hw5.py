#!/usr/bin/env python
"""
Train and predict using a Hidden Markov Model part-of-speech tagger.

Usage:
  hw5.py training_file test_file

Gail Godinez - fgodinez@bu.edu
Collaborative Partner - Kevin Amorim -kramorim@bu.edu

"""

import optparse
import collections
import math
import itertools

import hw5_common

# Smoothing methods
NO_SMOOTHING = 'None'  # Return 0 for the probability of unseen events
ADD_ONE_SMOOTHING = 'AddOne'  # Add a count of 1 for every possible event.
# *** Add additional smoothing methods here ***

# Unknown word handling methods
PREDICT_ZERO = 'None'  # Return 0 for the probability of unseen words

# If p is the most common part of speech in the training data,
# Pr(unknown word | p) = 1; Pr(unknown word | <anything else>) = 0
PREDICT_MOST_COMMON_PART_OF_SPEECH = 'MostCommonPos'
# *** Add additional unknown-word-handling methods here ***


class BaselineModel:
  '''A baseline part-of-speech tagger.

  Fields:
    dictionary: map from a word to the most common part-of-speech for that word.
    default: the most common overall part of speech.
  '''
  def __init__(self, training_data):
    '''Train a baseline most-common-part-of-speech classifier.

    Args:
      training_data: a list of pos, word pairs:
    '''
    # FIXME *** IMPLEMENT ME ***

    self.dictionary = None
    self.default = None

    if type(training_data) is dict:
      tagCount = {}
      wordD = {}
      for (tag, word) in training_data:
        if tag not in tagCount:
          tagCount[tag] = training_data[(tag, word)]
        else:
          tagCount[tag] += training_data[(tag, word)]
        if word not in wordD:
          wordD[word] = tag
        elif training_data[(tag, word)] > training_data[(wordD[word], word)]:
          wordD[word] = tag
      i = -9999999999999
      for tag in tagCount: #default tag for words
        if tagCount[tag] > i and tag !='<s>':
          i = tagCount[tag]
          self.default = tag
      self.dictionary = wordD

      ####
    
    else:
      tagTotal = {}
      dicct = {}
      pW = {}

      for (tag, word) in training_data:
        if tag in tagTotal: #counts the pos
          tagTotal[tag]+= 1.0
        else:
          tagTotal[tag] = 1.0
        if (tag, word) not in pW: #counts the (tag, word) pairs
          pW[(tag, word)] = 1.0
        else:
          pW[(tag, word)] += 1.0

      i = 0
      for tag in tagTotal: #default tag for words
        if tagTotal[tag] > i and tag !='<s>':
          i = tagTotal[tag]
          self.default = tag

      #finds largest pos given word and assigns accordingly
      for (tag, word) in pW:
        if word in dicct:
          if pW[(tag, word)] > pW[(dicct[word], word)]:
            i = pW[(tag, word)]
            dicct[word] = tag
        else:
          dicct[word] = tag
      self.dictionary = dicct

    

  def predict_sentence(self, sentence):
    return [self.dictionary.get(word, self.default) for word in sentence]


class HiddenMarkovModel:
  def __init__(self, order, emission, transition, parts_of_speech, known):
    baseline = BaselineModel(emission)
    # Order 0 -> unigram model, order 1 -> bigram, order 2 -> trigram, etc.
    self.order = order
    # Emission probabilities, a map from (pos, word) to Pr(word|pos)
    self.emission = emission
    # Transition probabilities
    # For a bigram model, a map from (pos0, pos1) to Pr(pos1|pos0)
    self.transition = transition
    # A set of parts of speech known by the model
    self.parts_of_speech = parts_of_speech
    # A set of words known by the model
    self.known_words = known
    # default pos
    self.baseline = baseline


  def predict_sentence(self, sentence):
    return self.find_best_path(self.compute_lattice(sentence))    

  def compute_lattice(self, sentence):
    """Compute the Viterbi lattice for an example sentence.

    Args:
      sentence: a list of words, not including the <s> tokens on either end.

    Returns:
      FOR ORDER 1 Markov models:
      lattice: [{pos: (score, prev_pos)}]
        That is, lattice[i][pos] = (score, prev_pos) where score is the
        log probability of the most likely pos/word sequence ending in word i
        having part-of-speech pos, and prev_pos is the part-of-speech of word i-1
        in that sequence.

        i=0 is the <s> token before the sentence
        i=1 is the first word of the sentence.
        len(lattice) = len(sentence) + 2.

      FOR ORDER 2 Markov models: ??? (extra credit)
    """
    # FIXME *** IMPLEMENT ME ***

    lattice = [] 
    lattice.append({'<s>': (math.log(1.0), None)}) #first part of sentence

    for i in range(0, len(sentence) + 1):
      if i == len(sentence): #when the sentence ends it makes the last node <s>
        thisW = '<s>'
      else:
        thisW = sentence[i] #gets part of sentence
      prevB = lattice[i]
      temp = {} #dictionary to be added to lattice
      if thisW not in self.known_words:
        for tag in self.parts_of_speech:
          if tag != '<s>':
            self.emission[(tag, thisW)] = 0
        self.emission[(self.baseline.default, thisW)] = math.log(1.0)
      for prev in prevB:
        for tag in self.parts_of_speech:
          if (tag, thisW) in self.emission:
            if (prev, tag) not in self.transition:
              self.transition[(prev, tag)] = 0
            maths = self.transition[(prev, tag)] + self.emission[(tag, thisW)]
            moreMath = lattice[i][prev][0] + maths
            if tag in temp: #if there is a previous node, adds the larger one to the dict
              if moreMath > temp[tag][0]:
                temp[tag] = ((moreMath), prev)
            else:
                temp[tag] = ((moreMath), prev)
            
      lattice.append(temp) #adds the dictionary for that word to lattice

    return lattice 



  @staticmethod
  def train(training_data,
      smoothing=ADD_ONE_SMOOTHING,
      unknown_handling=PREDICT_MOST_COMMON_PART_OF_SPEECH,
      order=1):
      # You can add additional keyword parameters here if you wish.
    '''Train a hidden-Markov-model part-of-speech tagger.

    Args:
      training_data: A list of pairs of a word and a part-of-speech.
      smoothing: The method to use for smoothing probabilities.
         Must be one of the _SMOOTHING constants above.
      unknown_handling: The method to use for handling unknown words.
         Must be one of the PREDICT_ constants above.
      order: The Markov order; the number of previous parts of speech to
        condition on in the transition probabilities.  A bigram model is order 1.

    Returns:
      A HiddenMarkovModel instance.

    
    '''

    
    transition = {}
    tcount = {} #transition count
    parts_of_speech = []
    known = []
    emission = {}
    allthecounts = {} #all counts for (pos,word)
    poscount = {}
    j = {}

    
    for i in range(0, len(training_data)):
      (pos, word) = training_data[i]
      if (pos, word) in allthecounts: #counting words w/ pos
        allthecounts[(pos, word)] += 1.0
      else:
        allthecounts[(pos, word)] = 1.0
      if pos in poscount: #counting pos
        poscount[pos] += 1.0
      else:
        poscount[pos] = 1.0
      if pos not in parts_of_speech: #makes list of PoS
        parts_of_speech.append(pos)
      if word not in known: #makes list of words
        known.append(word)
      if i < len(training_data)-1:
        q = training_data[i][0] #first one
        p = training_data[i+1][0] #next one
        if (q, p) not in tcount:
          tcount[(q, p)] = 1.0 #adds to list if not there
        else:
          tcount[(q, p)]+= 1.0 #adds +1 to each transition's counter
        if q in j:
          j[q] += 1.0
        else:
          j[q] = 1.0


    #Add One Smoothing
    if smoothing == ADD_ONE_SMOOTHING:
      for (x, y) in set(itertools.product(parts_of_speech,known)):
        if (x, y) in allthecounts:
          allthecounts[(x, y)] += 1.0
        else:
          allthecounts[(x, y)] = 1.0
        if x in poscount:
          poscount[x] += 1.0
      for (x, y) in set(itertools.product(parts_of_speech,parts_of_speech)): #these two lines tho
        if (x, y) in tcount:
          tcount[(x, y)] += 1.0
        else:
          tcount[(x, y)] = 1.0
        if x in j:
          j[x] += 1.0
        else:
          j[x] = 1.0
    
    #Calculates emission probabilities
    for (pos, word) in allthecounts:
      count = allthecounts[(pos, word)]
      posC = poscount[pos]
      if count/posC < 0.000001:
        emission[(pos, word)] = float('-inf')
      else:
        emission[(pos, word)] = math.log(count/posC) #probability that a word has PoS pos

    #Caculates transmission probabilities
    for (x,y) in tcount:
      maths = tcount[(x, y)]/j[x]
      if maths < 0.000001:
        transition[(x, y)] = float('-inf')
      else: 
        transition[(x, y)] = math.log(maths)

    return HiddenMarkovModel(order, emission, transition, set(parts_of_speech), set(known))

 
  @staticmethod
  def find_best_path(lattice):
    """Return the best path backwards through a complete Viterbi lattice.

    Args:
      FOR ORDER 1 MARKOV MODELS (bigram):
        lattice: [{pos: (score, prev_pos)}].  See compute_lattice for details.

    Returns:
      FOR ORDER 1 MARKOV MODELS (bigram):
        A list of parts of speech.  Does not include the <s> tokens surrounding
        the sentence, so the length of the return value is 2 less than the length
        of the lattice.
    """

    best = ['POS']*(len(lattice)) #creates 'empty' list for best path
    i = len(lattice)-1 #index to go backwards from lattice
    prev = '<s>' #initaliztion of prev word

    while i > -1:
      word = lattice[i] #grabs the dictionary
      for pos in word: #grabs part of speech in dictionary
        if pos == prev and best[i] == 'POS': #looks for next best POS from prev
          best[i] = pos
          prev = word[pos][1]
      i-=1

    #deletes <s> at beginning and end of sentence
    del best[0]
    del best[len(best)-1]

    return best

  


def main():
  parser = optparse.OptionParser()
  parser.add_option('-s', '--smoothing', choices=(NO_SMOOTHING,
    ADD_ONE_SMOOTHING), default=NO_SMOOTHING)
  parser.add_option('-o', '--order', default=1, type=int)
  parser.add_option('-u', '--unknown',
      choices=(PREDICT_ZERO, PREDICT_MOST_COMMON_PART_OF_SPEECH,),
      default=PREDICT_ZERO)
  options, args = parser.parse_args()
  train_filename, test_filename = args
  training_data = hw5_common.read_part_of_speech_file(train_filename)
  if options.order == 0:
    model = BaselineModel(training_data)
  else:
    model = HiddenMarkovModel.train(
        training_data, options.smoothing, options.unknown, options.order)
  predictions = hw5_common.get_predictions(
      test_filename, model.predict_sentence)
  for word, prediction, true_pos in predictions:
    print word, prediction, true_pos

if __name__ == '__main__':
  main()
