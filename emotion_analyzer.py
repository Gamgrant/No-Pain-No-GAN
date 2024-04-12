from hume import HumeBatchClient
from hume.models.config import LanguageConfig, FaceConfig
import json
import matplotlib.pyplot as plt
from vault_secrets import HUME_API_KEY

class EmotionAnalyzer:
    def __init__(self, api_key):
        self.client = HumeBatchClient(api_key)
        self.lang_config = LanguageConfig()

    @staticmethod
    def print_json_structure(json_obj, indent=0):
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if isinstance(value, dict):
                    print('  ' * indent + str(key) + ' - Dict{' + str(len(value)) + '}:')
                    EmotionAnalyzer.print_json_structure(value, indent + 1)
                elif isinstance(value, list):
                    if value:  
                        print('  ' * indent + str(key) + ' - List[' + str(len(value)) + ']:')
                        if isinstance(value[0], dict):
                            print('  ' * (indent + 1) + 'Dict{' + str(len(value[0])) + '}:')
                            EmotionAnalyzer.print_json_structure(value[0], indent + 1)
                    else:
                        print('  ' * indent + str(key) + ' - List[0]')
                else:
                    print('  ' * indent + str(key) + ': ' + type(value).__name__)
        else:
            print('  ' * indent + type(json_obj).__name__)

    # only used for plotting emotions
    def process_model_data(self, predictions, emotion_mapping):
        model_data = []
        for prediction in predictions:
            if 'frame' in prediction:
                time = prediction.get('time', 0)
            else:
                begin_timestamp = prediction.get('time', {}).get('begin', 0)
                end_timestamp = prediction.get('time', {}).get('end', 0)
                time = begin_timestamp+(end_timestamp-begin_timestamp)/2
            emotions = prediction.get('emotions', [])
            if emotions:
                highest_emotion = max(emotions, key=lambda e: e.get('score', 0))
                y_value = emotion_mapping.setdefault(highest_emotion['name'], len(emotion_mapping))
                model_data.append((time, y_value))
        return model_data

    def plot_emotions_flex(self, song_response):
        emotion_mapping = {}
    
        face_data = []
        language_data = []
        
        # check availability of face and language response 
        if 'face' in song_response['results']['predictions'][0]['models']:
            face_data = song_response['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions']
        
        if 'language' in song_response['results']['predictions'][0]['models']:
            language_data = song_response['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions']

        face_emotions = self.process_model_data(face_data, emotion_mapping) if face_data else []
        language_emotions = self.process_model_data(language_data, emotion_mapping) if language_data else []

        # Separate times and emotions for plotting
        face_times, face_y = zip(*face_emotions) if face_emotions else ([], [])
        lang_times, lang_y = zip(*language_emotions) if language_emotions else ([], [])

        plt.figure(figsize=(15, 6))
        if face_emotions:
            plt.plot(face_times, face_y, color='blue', label='Face Model')
        if language_emotions:
            plt.plot(lang_times, lang_y, color='red', label='Language Model')

        # Plot config 
        plt.xlabel('Time (s)')
        plt.ylabel('Emotion')
        plt.yticks(range(len(emotion_mapping)), list(emotion_mapping.keys()), rotation=45) 
        plt.title('Emotion Path')
        plt.legend()
        plt.tight_layout()  
        plt.show()
    
    # return top k emotions for each song
    def batch_inference(self, files, k):
        job = self.client.submit_job([], [self.lang_config], files=files)
        print("Running...", job)
        job.await_complete()
        response = job.get_predictions()

        results = []
        for song_response in response:
            # Initialize a dictionary to count the frequency of each emotion
            emotion_counts = {}

            # Extract predictions from the response
            predictions = song_response['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions']
            for prediction in predictions:
                # Find the emotion with the highest score at this timestamp/prediction
                if prediction['emotions']:  # Ensure there is at least one emotion
                    highest_emotion = max(prediction['emotions'], key=lambda e: e['score'])
                    emotion_name = highest_emotion['name']
                    
                    # Increment the count for this emotion
                    if emotion_name in emotion_counts:
                        emotion_counts[emotion_name] += 1
                    else:
                        emotion_counts[emotion_name] = 1

            # Sort the emotions based on frequency and get the top k
            top_emotions = sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True)[:k]
            
            # Append the top emotions for this song, along with the song name, to the results list
            song_name = song_response['source']['filename']
            results.append((song_name, top_emotions))
        
        return results

    # main function used in main but mainly (no pun intended) for testing
    def run_emotion_analysis(self, files):
        results = self.batch_inference(files, 3)
        print(results)

def main():
    analyzer = EmotionAnalyzer(HUME_API_KEY)
    files = ["SAS_Rant.mp4", "heard_em_say_kanye.mp3"]
    analyzer.run_analysis(files)

if __name__ == "__main__":
    main()
