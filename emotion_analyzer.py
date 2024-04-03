from hume import HumeBatchClient
from hume.models.config import LanguageConfig, FaceConfig
import json
import matplotlib.pyplot as plt
from vault_secrets import API_KEY

def print_json_structure(json_obj, indent=0):
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, dict):
                print('  ' * indent + str(key) + ' - Dict{' + str(len(value)) + '}:')
                print_json_structure(value, indent + 1)
            elif isinstance(value, list):
                if value:  
                    print('  ' * indent + str(key) + ' - List[' + str(len(value)) + ']:')
                    if isinstance(value[0], dict):
                        print('  ' * (indent + 1) + 'Dict{' + str(len(value[0])) + '}:')
                        print_json_structure(value[0], indent + 1)
                else:
                    print('  ' * indent + str(key) + ' - List[0]')
            else:
                print('  ' * indent + str(key) + ': ' + type(value).__name__)
    else:
        print('  ' * indent + type(json_obj).__name__)

def process_model_data(predictions, emotion_mapping):
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

def plot_emotions_flex(json_data):
    emotion_mapping = {}
    
    face_data = []
    language_data = []
    
    # check availability of face and language response 
    if 'face' in json_data['results']['predictions'][0]['models']:
        face_data = json_data['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions']
    
    if 'language' in json_data['results']['predictions'][0]['models']:
        language_data = json_data['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions']

    face_emotions = process_model_data(face_data, emotion_mapping) if face_data else []
    language_emotions = process_model_data(language_data, emotion_mapping) if language_data else []

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

def main():
    # pass in files 
    files = ["SAS_Rant.mp4"]

    # init Hume client
    client = HumeBatchClient(API_KEY)
    lang_config = LanguageConfig()
    face_config = FaceConfig()

    # run job 
    job = client.submit_job([], [face_config, lang_config], files=files)
    print("Running...", job)
    job.await_complete()
    predictions = job.get_predictions()
    
    # analyze response and extract preds to generate emotion path 
    predictions_temp = predictions.copy()
    preds = predictions_temp.pop()
    print_json_structure(preds)
    plot_emotions_flex(preds)

if __name__ == "__main__":
    main()