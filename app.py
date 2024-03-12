from typing import Dict
from pydub import silence, AudioSegment
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import io
from sklearn.feature_extraction.text import CountVectorizer
import operator
import warnings
from pydantic import BaseModel
from mangum import Mangum
import subprocess
import numpy as np
import joblib
import tempfile
import random
import requests
import soundfile as sf
#from scipy.signal import stft
from python_speech_features import mfcc
#import requests
import textstat
warnings.filterwarnings("ignore")
#from spellchecker import SpellChecker
from scipy.signal import stft
from textblob import TextBlob

#os.environ['LIBROSA_CACHE_DIR'] = '/usr/'
#librosa.cache.path = '/tmp/librosa_cache'


app = FastAPI()
handler = Mangum(app)

####2. Syllable count and Speech rate calculation
def syllable_count(sentence):
    words = sentence.split()
    total_syllables = 0
    for word in words:
        total_syllables += count_syllables_in_word(word)
    return total_syllables

def count_syllables_in_word(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

#2. Speech Rate (SR), calculated by dividing total number of syllables uttered by the total number of minutes of recording time, including pauses.
def speech_rate(syllables_uttered, total_time_in_seconds):
    if total_time_in_seconds<60:
        total_time_in_minutes = 1
    else:
        total_time_in_minutes = total_time_in_seconds/60

    SR = syllables_uttered / total_time_in_minutes
    if SR>=180:
        reason = "Fast Speech Rate"
    elif SR>150 and SR<180:
        reason = "Medium Speech Rate"
    elif SR<150:
        reason = "Slow Speech Rate"
    return SR, reason

####3. Articulation rate and Pauses in speech to be detected and 
####4. PTR to get pause frequency
##Articulation Rate (AR), calculated by dividing total number of syllables uttered by the minutes of speech time only, excluding pauses
def detect_pauses(audio, total_duration_of_audio):
    AudioSegment.converter = "/usr/share/ffmpeg"
    segment = AudioSegment.from_wav(io.BytesIO(audio))
    print(3)
    pause_time_milliseconds = 0
    print(3)
    silence_ranges = silence.detect_silence(segment, min_silence_len= 500 ,silence_thresh=-32.0)
    print(3)
    for i in silence_ranges:
        pause_time_milliseconds += i[1] - i[0]
    pause_time_seconds = (pause_time_milliseconds/1000)
    num_pauses = len(silence_ranges)
    if total_duration_of_audio <60:
        total_duration_of_audio = 1
    else:
        total_duration_of_audio = round(total_duration_of_audio/60)
    pause_frequency = num_pauses / total_duration_of_audio

    return pause_time_seconds, pause_frequency

def articulation_rate(time_without_pauses, syllables_uttered):
    if time_without_pauses<60:
        time_in_minutes = 1
    else:
        time_in_minutes = time_without_pauses/60
    return syllables_uttered/time_in_minutes
    
###5. BAG of words

def most_repeated_words(text):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't"]
    vectorizer = CountVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense_array = X.toarray()
    word_frequencies = dict(zip(feature_names, dense_array.sum(axis=0)))
    sorted_word_frequencies = dict(sorted(word_frequencies.items(), key=operator.itemgetter(1), reverse=True))
    top_3_words = list(sorted_word_frequencies.items())[:3]
    top_3_words_string = ', '.join([word for word, count in top_3_words])
    return top_3_words_string
###6. Fluency

def spectral_flux_1(audio_data, sample_rate, window_size=2048, hop_size=512):
    # Compute Short-Time Fourier Transform (STFT)
    _, _, Zxx = stft(audio_data, fs=sample_rate, nperseg=window_size, noverlap=window_size-hop_size)

    # Calculate magnitude spectrogram
    magnitude_spec = np.abs(Zxx)

    # Compute spectral flux
    spectral_flux_values = np.sqrt(np.sum(np.diff(magnitude_spec, axis=1)**2, axis=1))

    return np.mean(spectral_flux_values)

def zero_crossing_rate(audio_data, frame_length=2048, hop_length=512):
    num_frames = (len(audio_data) - frame_length) // hop_length + 1
    zcr_values = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio_data[start:end]
        num_crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr_values[i] = num_crossings / (len(frame) - 1)

    return np.mean(zcr_values)

def Get_Fluency(file_name):
    n_mfccs = 20 # This variable is tunneable with each run
    number_of_features = 3 + n_mfccs
    #number_of_features = 154 + n_mfccs # 154 are the total values returned by rest of computed features
    features= np.empty((0,number_of_features))
    print("Loading Features")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(file_name)  # Wr
    print("Stored temp file")
    X, sample_rate = sf.read(temp_file.name)
    print("Librosa Load")
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    mfccs = np.mean(mfcc(X, samplerate=sample_rate, numcep=20), axis=0)
    print("mfccs")
    rmse = np.sqrt(np.mean(X**2, axis=0))
    print("rmse")
    spectral_flux = spectral_flux_1(X, sample_rate)
    print("spectral_flux")
    #zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0) #Returns 1 value
    zcr = zero_crossing_rate(X)
    print("ZCR")

    extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
    print("Extracted_features")
    #print "Total Extracted Features: ", len(extracted_features) #This helps us identify really how many features are being computed
    features = np.vstack([features, extracted_features])
    loaded_svm_model = joblib.load('/usr/share/svm_model.pkl')
    #loaded_svm_model = joblib.load('./svm_model.pkl')
    print("Load Model")
    loaded_scaler = joblib.load('/usr/share/standard_scaler.pkl')
    #loaded_scaler = joblib.load('./standard_scaler.pkl')
    print("Load scaler")
    new_data_scaled = loaded_scaler.transform(np.array(features).astype(np.float32))
    new_predictions = loaded_svm_model.predict(new_data_scaled)
    new_predictions_proba = loaded_svm_model.predict_proba(new_data_scaled)
    max_proba =  max(new_predictions_proba[0])
    print("Predicted", max_proba)
    fluency = ['Low','Intermediate','High']
    fluency_output = fluency[new_predictions[0]]
    #print(new_predictions, fluency_output)
    if fluency_output == "Low":
        fluency_score = max_proba * 0.3
    if fluency_output == "Intermediate":
        fluency_score = max_proba * 0.65
    if fluency_output == "High":
        fluency_score = max_proba * 1
    #new_probabilities = loaded_svm_model.predict_proba(new_data_scaled)
    
    return fluency, fluency_score

###7. Spelling Errors
'''def get_spelling_errors(sentence):
    spell = SpellChecker()
    words = sentence.split()  
    spelling_errors = spell.unknown(words)
    return list(spelling_errors)'''


###8.Mean Length of Runs (MLR): The mean number of syllables uttered between hesitations. It indicates the length of utterances between pauses. 
def calculate_mlr(syllables_uttered, total_time_in_seconds, pause_time_in_seconds):
    print(syllables_uttered - 1)
    print(total_time_in_seconds - pause_time_in_seconds)
    mlr = (syllables_uttered - 1) / (total_time_in_seconds - pause_time_in_seconds)
    return mlr

def get_clarity(text):
    value = textstat.flesch_reading_ease(text)
    print("Redability calculated")
    if value>90:
        Clarity_text = "Very Easy to Understand"
        clarity = "High"
        if value >100:
            value = 100
    if value >70 and value <=90:
        Clarity_text = "Easy to read. Conversational English for consumers."
        clarity = "High"
    if value >50 and value <=70:
        Clarity_text = "Fairly easy to read or Fairly easy to understand"
        clarity = "Intermediate"
    if value >30 and value <=50:
        Clarity_text = "Hard to understand"
        clarity = "Intermediate"
    if value <=30:
        Clarity_text = "Very hard to understand"
        clarity = "Low"
        if value <0:
            value = 0
    return clarity, Clarity_text, value


def get_confidence(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity

    # You can define your confidence thresholds based on sentiment polarity
    if sentiment_polarity > 0.5:
        confidence_level = "High"
        confidence_score = round(sentiment_polarity*100)
    elif 0 <= sentiment_polarity <= 0.5:
        confidence_level = "Medium"
        confidence_score = round(sentiment_polarity*100)

    else:
        confidence_score = round(sentiment_polarity*100)
        confidence_level = "Low"

    return confidence_level, confidence_score

def recommendations(SR_reason, AR, PTR, MLR, Fluency, Fluency_proba, Clarity,  Clarity_text, Clarity_score, confidence, confidence_score):
    dict_recom = {}
    Area_of_Improvement = []
    Actionable_Recommendations = []
    print("Recommendations start")
    if SR_reason == "Slow Speech Rate" or SR_reason == "Fast Speech Rate":
        Area_of_Improvement .append("Clarity Based on Speech rate")
        Actionable_Recommendations.append("As you are a "+SR_reason+". Practice adjusting speech rate for better clarity. Use pauses effectively") 
    if AR<20:
        Area_of_Improvement.append("Articulation Rate")
        Actionable_Recommendations.append("Improve articulation by repeating certain phrases and read out loud to increase clarity")
    if Fluency_proba <50:
        Area_of_Improvement.append("Fluency")
        fluency_list = ["Start by slowing down your speech. Speaking too quickly can contribute to disfluencies. Focus on articulating each word clearly.",
                        "Engage in tongue twisters to enhance your tongue and lip coordination. These exercises can be fun and effective for improving fluency.",
                        "Incorporate intentional pauses in your speech. Pausing can provide time to gather your thoughts and promote smoother speech.",
                        "Clear articulation contributes to overall fluency. Focus on pronouncing each word accurately to improve the smoothness of your speech."]
        Actionable_Recommendations.append(random.choice(fluency_list))
    if Clarity_score<50:
        Area_of_Improvement.append("Clarity on content")
        clarity_list = ["Remove unnecessary repetition and redundant phrases.", "Clearly structure your content with a logical flow.",
                        "Keep sentences clear and concise; avoid unnecessary complexity.",
                        "Employ transition words to guide your audience through the logical progression of your ideas."]
        Actionable_Recommendations.append(random.choice(clarity_list))
    if confidence_score< 50:
        Area_of_Improvement.append("Confidence")
        confidence_list = ["Avoid ambiguity and vagueness", "Ensure a solid understanding of the topic through comprehensive research.",
                           "Minimize the use of repetative phrases", "Reduce the pauses during speech"]
        Actionable_Recommendations.append(random.choice(confidence_list))
    print("Adding values to dictonary in recommendations")
    for i in range(len(Area_of_Improvement)):
        dict_recom[Area_of_Improvement[i]] = Actionable_Recommendations[i]
    
    dict_recom = {k:dict_recom[k] for k in random.sample(list(dict_recom.keys()), len(dict_recom))}
    print("Dict shuffle", type(dict_recom))
    return dict_recom

def check_grammar(text):
    # LanguageTool API endpoint
    api_url = "https://languagetool.org/api/v2/check"

    # Specify language (e.g., English)
    language = "en-US"

    # Prepare data for the POST request
    data = {
        "language": language,
        "text": text
    }

    # Make the POST request
    response = requests.post(api_url, data=data)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()

        # matches = tool.check(interview.text_data)
        grammar_score = 1 - len(result.get("matches", [])) / len(text.split())
        return round(grammar_score*100)

    else:
        print(f"Error: Unable to check grammar. Status code: {response.status_code}")


class Interview(BaseModel):
    text_data: str
    stream_url: str
    

@app.post("/analyze")
async def analyze_data(interview:Interview):
    print(1)
    try:
        command = [
            '/usr/share/ffmpeg',
             #'ffmpeg',
            '-i',
            interview.stream_url,
            '-b:a', '64k',
            '-f', 'wav',  # Force output format to WAV
            'pipe:1'  # Send output to stdout
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stderr:
            stderr_text = stderr.decode('utf-8')
            duration_line = [line for line in stderr_text.split('\n') if 'Duration' in line][0]
            duration = duration_line.split(' ')[3]
            hours, minutes, seconds = map(float, duration[:-1].split(':'))
            total_time_in_seconds = hours * 3600 + minutes * 60 + seconds
        else:
            print("No duration information found in stderr.")
        #matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(), re.DOTALL).groupdict()
        audio_file_path = stdout
        text_data = interview.text_data
        syllables_uttered = syllable_count(text_data)
        pause_time_in_seconds, pause_frequency_minutes = detect_pauses(audio_file_path, total_time_in_seconds)
        #Big5_traits = Personality_Detection_from_reviews_submitted(text_data)
        SR, SR_Reason = speech_rate(syllables_uttered, total_time_in_seconds)
        AR = articulation_rate(total_time_in_seconds-pause_time_in_seconds, syllables_uttered)
        print("AR")
        PTR = pause_frequency_minutes
        print("PTR")
        MLR  = calculate_mlr(syllables_uttered, total_time_in_seconds,pause_time_in_seconds)
        print("MLR")
        Most_repeated_words = most_repeated_words(text_data)
        print("Most repeated words")
        Fluency, Fluency_proba = Get_Fluency(audio_file_path)
        print("Fluency")
        Clarity ,  Clarity_text, Clarity_score = get_clarity(text_data)
        print("Clarity")
        confidence, confidence_score = get_confidence(text_data)
        print("Confidence")
        recommendation_dict = recommendations(SR_Reason, AR, PTR, MLR, Fluency, Fluency_proba, Clarity,  
                                              Clarity_text, Clarity_score, confidence, confidence_score)
        print("Recommendations")
        Grammer_score = check_grammar(text_data)
        print("Grammer Score")
        Area_of_Improvements = []
        Actionable_Recommendations = []
        for key, value in recommendation_dict.items():
            Area_of_Improvements.append(key)
            Actionable_Recommendations.append(value)

        result = {"Speech_rate": round(SR,2), "Articulation_Rate":round(AR,2), "Phonation_or_Time_Ratio": round(PTR,2),
                "Mean_Length_of_runs": round(MLR,2), "Fluency_and_Coherence": round(Fluency_proba*100), "Clarity" : Clarity_score,
                "Confidence": confidence_score, "Grammer_score": Grammer_score,
                "Area_of_Improvements":Area_of_Improvements ,
                "Actionable_Recommendations": Actionable_Recommendations}

        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}") 

@app.get("/testing")
async def testing():
    return {"testing": "testing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)