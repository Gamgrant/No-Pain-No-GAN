from hume import HumeBatchClient, TranscriptionConfig
from hume.models.config import LanguageConfig, FaceConfig, ProsodyConfig, BurstConfig
import json
import matplotlib.pyplot as plt
from vault_secrets import HUME_API_KEYS

class EmotionAnalyzer:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client = HumeBatchClient(self.api_keys[self.current_key_index])
        self.model_configs = [LanguageConfig()]
        self.transcription_config = TranscriptionConfig(language="en")
        self.error_record = {}
    
    def update_api_key(self):
        """Rotate to the next available API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = HumeBatchClient(self.api_keys[self.current_key_index])
        print(f"API Key rotated to: {self.api_keys[self.current_key_index]}")

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
        prosody_data = []
        
        # check availability of face and language response 
        if 'face' in song_response['results']['predictions'][0]['models']:
            face_data = song_response['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions']
        
        if 'language' in song_response['results']['predictions'][0]['models']:
            language_data = song_response['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions']
        if 'prosody' in song_response['results']['predictions'][0]['models']:
            prosody_data = song_response['results']['predictions'][0]['models']['prosody']['grouped_predictions'][0]['predictions']

        face_emotions = self.process_model_data(face_data, emotion_mapping) if face_data else []
        language_emotions = self.process_model_data(language_data, emotion_mapping) if language_data else []
        prosody_emotions = self.process_model_data(prosody_data, emotion_mapping) if prosody_data else []

        # Separate times and emotions for plotting
        face_times, face_y = zip(*face_emotions) if face_emotions else ([], [])
        lang_times, lang_y = zip(*language_emotions) if language_emotions else ([], [])
        prosody_times, prosody_y = zip(*prosody_emotions) if prosody_emotions else ([], [])

        plt.figure(figsize=(15, 6))
        if face_emotions:
            plt.plot(face_times, face_y, color='blue', label='Face Model')
        if language_emotions:
            plt.plot(lang_times, lang_y, color='red', label='Language Model')
        if prosody_emotions:
            plt.plot(prosody_times, prosody_y, color='green', label='Prosody Model')

        # Plot config 
        plt.xlabel('Time (s)')
        plt.ylabel('Emotion')
        plt.yticks(range(len(emotion_mapping)), list(emotion_mapping.keys()), rotation=45) 
        plt.title('Emotion Path')
        plt.legend()
        plt.tight_layout()  
        plt.show()
    
    # return top k emotions for each song
    def batch_inference(self, files, k, retry_count=0, debug=True):
        job = self.client.submit_job([], self.model_configs, files=files)
        print("Running...", job)
        try:
            job.await_complete()
            response = job.get_predictions()
            if debug:
                with open('debug_results.json', 'w') as f:
                    json.dump(response, f)
            return self.parse_results(response, k)
        except Exception as e:
            if "E0300" in str(e) and retry_count < len(self.api_keys):  # Assuming E0300 is the error code for rate limits or key issues
                print("API rate limit exceeded, rotating key.")
                self.update_api_key()
                return self.batch_inference(files, k, retry_count + 1)  # Retry with a new key
            raise e

    def parse_results(self, response, k):
        results = []
        for song_response in response:
            emotion_counts = {}
            song_name = song_response['source']['filename'].split('.')[0]
            if not song_response['results']['predictions'] or song_response['results']["errors"]:
                self.error_record[song_name] = song_response['results']['errors']
                    # print(f"Error for {song_name}: {song_response['results']['errors']}")
                results.append((song_name, [("N/A", 0)]))
                continue
            try:
                language_predictions = song_response['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions']
            except:
                print(f"Error for {song_name}: {song_response['results']}")
                self.error_record[song_name] = song_response['results']
                continue
            for prediction in language_predictions:
                if prediction['emotions']:
                    highest_emotion = max(prediction['emotions'], key=lambda e: e['score'])
                    emotion_name = highest_emotion['name']
                    emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
            
            top_emotions = sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True)[:k]
            results.append((song_name, top_emotions))
        return results

    def run_emotion_analysis(self, files, k):
        results = self.batch_inference(files, k)
        return results

def main():
    analyzer = EmotionAnalyzer(HUME_API_KEYS)
    files = ["water_tyla.mp3", "SAS_Rant.mp4", "Heard_Em_Say_kanye.mp3"]
    results = analyzer.run_emotion_analysis(files, 3)
    print(results)
   

if __name__ == "__main__":
    main()
