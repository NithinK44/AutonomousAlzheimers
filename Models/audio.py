from gtts import gTTS 
import os 
from audioplayer import AudioPlayer

# Playback stops when the object is destroyed (GC'ed), so save a reference to the object for non-blocking playback.

def say(data):
    language = 'en'
      
    myobj = gTTS(text=str(data), lang=language, slow=False) 

    myobj.save("Models/text.mp3") 

    #os.system("start text.mp3")
    AudioPlayer("Models/text.mp3").play(block=True)

#say("hi")
