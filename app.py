print("hello world")

import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        voice = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(voice)
        print("You said:", text)
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
    
speak("hello world!")
listen()
