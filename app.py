import speech_recognition as sr
import pyttsx3

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        voice = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(voice)
        # print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio!")
        return None
    except sr.RequestError as e:
        print("Request failed; {0}".format(e))
        return None

def speak(text):
    engine.say(text)
    engine.runAndWait()
    
# USING THE SAVED MODEL:
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("./src/intents.json").read())

words = pickle.load(open("./src/words.pkl", "rb"))
classes = pickle.load(open("./src/classes.pkl", "rb"))

model = load_model("./src/model.h5")


def cleanup_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence.lower())
  return [lemmatizer.lemmatize(word) for word in sentence_words]

def bag_of_words(sentence):
  sentence_words = cleanup_sentence(sentence)
  bag = [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  X = bag_of_words(sentence)
  y = model.predict(np.array([X]), verbose=0)[0]
  ERROR_THRESHOLD = 0.25
  results = [[i, r] for i, r in enumerate(y) if r > ERROR_THRESHOLD]

  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
  return return_list

def get_response(intents_list, intents_json):
  if not len(intents_list): 
    return "<No Response. ERROR: Unknown Class>"
	
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i["tag"] == tag:
      return random.choice(i['responses'])


print("GBot is running...")
while True:
  # message = input("You: ")
  message = listen()
  while(not message):
    message = listen()
    
  print("You:", message)
  ints = predict_class(message)
  # print(ints)
  response = get_response(ints, intents)
    
  print("GBot:", response)
  speak(response)
  message = ""
    
# speak("hello world!")
# listen()
