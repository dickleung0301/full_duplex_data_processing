import json
import pandas as pd
from datasets import Dataset, load_dataset

def download_huggingface_librispeech(split: str) -> Dataset:
    """
    a function to download librispeech
    """

    return load_dataset("openslr/librispeech_asr", split=split)

def convert_2_huggingface_path(
    path_of_librispeech: str,
    path_of_huggingface: str,
    output_path: str,
    split: str
):
    """
    a function to convert the path to that of huggingface
    """

    # instantiate lists to store the context and audio path
    context = []
    audio_path = []
    output_data = []

    # download librispeech from huggingface
    huggingface_librispeech = download_huggingface_librispeech(split=split)
    original_context = huggingface_librispeech["text"].tolist()

    # load the json file of librispeech from original site
    with open(path_of_librispeech, "r", encoding="utf-8") as f:

        librispeech_path_json = json.load(f)

    # load the json file of librispeech from huggingface
    with open(path_of_huggingface, "r", encoding="utf-8") as f:

        huggingface_path_json = json.load(f)

    # extract the context and audio path
    for datum in huggingface_path_json:

        context.append(datum["messages"][2]["content"])
        audio_path.append(datum["audios"][0])

    # make a spreadsheet for matching the paths
    huggingface_df = pd.DataFrame({
        "audio_path": audio_path,
        "context": context
    })

    # make another spreadsheet for context and also id
    huggingface_librispeech = pd.DataFrame({
        "id": huggingface_librispeech["id"].tolist(),
        "context": [text.lower().strip() for text in original_context]
    })
    
    # merge with unique key
    merged_df = pd.merge(huggingface_librispeech, huggingface_df, on="context", how="left")

    for datum in librispeech_path_json:

        matches = merged_df.loc[merged_df["id"] == datum["audios"][0].split("/")[-1].split(".flac")[0].strip(), "audio_path"].tolist()

        # check if we found anything
        if matches:
            
            datum["audios"][0] = matches[0]
            output_data.append(datum)

    # save the json file
    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(librispeech_path_json, f, indent=4, ensure_ascii=False)