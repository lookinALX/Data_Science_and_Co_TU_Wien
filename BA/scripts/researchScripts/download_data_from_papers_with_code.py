import requests


def get_dataset_info(dataset_name):
    base_url = "https://paperswithcode.com/api/v1/datasets/"
    search_url = f"https://paperswithcode.com/api/v1/datasets/?q={dataset_name}"

    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            index = next((i for i, result in enumerate(data["results"]) if result["name"] == dataset_name), None)
            if index is not None:
                dataset_id = data["results"][index]["id"]
                dataset_info = requests.get(f"{base_url}{dataset_id}/").json()
                return dataset_info
        else:
            return {"error": "Dataset not found"}
    else:
        return {"error": "API request failed"}

def get_dataset_evaluations(dataset_id):
    url = f"https://paperswithcode.com/api/v1/datasets/{dataset_id}/evaluations/"
    evaluations_ids = []

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            for i in data['results']:
                evaluations_ids.append(i['id'])
            return evaluations_ids
        else:
            return []
    else:
        return {"error": "API request failed"}


def get_number_of_evaluations(evaluation_ids):
    count = 0
    if len(evaluation_ids) == 0:
        return 0
    for i in evaluation_ids:
        response = requests.get(f"https://paperswithcode.com/api/v1/evaluations/{i}/results/")
        if response.status_code == 200:
            data = response.json()
            count += int(data['count'])
        else:
            return {"error": "API request failed"}
    return count

if __name__ == "__main__":
    datasets = [
        "AudioSet", "AudioCaps", "AVSpeech", "UrbanSound8K", "ESC-50", "LibriSpeech",
        "VGG-Sound", "FSD50K", "Common Voice", "GTZAN", "VoxCeleb1", "ICBHI Respiratory Sound Database",
        "SHD", "Speech Commands", "TAU-NIGENS Spatial Sound Events 2020", "TUT Sound Events 2017",
        "MACS (Multi-Annotator Captioned Soundscapes)", "MagnaTagATune",
        "Kinetics-700", "MusicNet", "EmoDB Dataset (Berlin Database of Emotional Speech)", "SEP-28k", "EPIC-SOUNDS",
        "MINDS-14", "BGG dataset (PUBG Gun Sound Dataset)", "ReefSet", "Multimodal PISA", "WavText5k", "OpenMIC-2018",
        "MELD", "MedleyDB", "FLEURS", "Audio Dialogues", "DEEP-VOICE: DeepFake Voice Recognition (Jordan Bird)",
        "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)", "SoundDesc", "Switchboard-1 Corpus",
        "MUSDB18", "DiCOVA", "IEMOCAP", "WSJ0-2mix", "DEMAND", "Coswara Dataset", "WavCaps", "TED-LIUM", "COUGHVID",
        "WHAM! (WSJ0 Hipster Ambient Mixtures)", "Cat Meow", "CochlScene", "SONYC-UST-V2"
    ]

    all_datasets_info = {}
    for dataset in datasets:
        print(f"Fetching info for: {dataset}")
        dataset_info = get_dataset_info(dataset)
        if (dataset_info is not None and not 'error' in dataset_info):
            dataset_evaluations_ids = get_dataset_evaluations(dataset_info['id'])
            dataset_number_of_evaluations = get_number_of_evaluations(dataset_evaluations_ids)
            all_datasets_info[dataset] = {'name': dataset_info['name'],
                                          'url': dataset_info['url'],
                                          'evaluations number': dataset_number_of_evaluations}
        else:
            all_datasets_info[dataset] = {'name': dataset,
                                          'url': '',
                                          'evaluations number': ''}

    for dataset_name, dataset_info in all_datasets_info.items():
        print(f"Dataset: {dataset_name}")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        print("\n")

    with open("datasets_info.txt", "w", encoding="utf-8") as file:
        for dataset_name, dataset_info in all_datasets_info.items():
            file.write(f"Dataset: {dataset_name}\n")
            for key, value in dataset_info.items():
                file.write(f"  {key}: {value}\n")
            file.write("\n")