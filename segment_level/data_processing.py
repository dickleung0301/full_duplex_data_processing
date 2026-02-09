import re
import json
import string
import pandas as pd
from .download_data_from_hf import(
    download_forced_aligned_librispeech,
    download_huggingface_librispeech
)

def read_json_line_file(json_file_path: str) -> list[dict]:
    """
    a function to read the annotated json file of librispeech-pc
    """

    # instantiate a list to store each json entry
    json_data = []

    # read the json file
    with open(json_file_path, "r", encoding="utf-8") as f:

        # loop through the lines
        for line in f:

            # skip the line that is empty
            if line.strip():

                json_data.append(json.loads(line))

    return json_data

def split_sentence_wrt_punctuation(text: str) -> list[str]:
    """
    a function to split the text line w.r.t. punctuation
    """

    # split the text once there is a punctuation
    segmantic_level_segments = re.split(r'[!?,.]', text)

    # filter out the edge case
    segmantic_level_segments = [segment.strip() for segment in segmantic_level_segments if segment.strip()]

    return segmantic_level_segments

def split_sentence_retaining_punctuation(text: str) -> list[str]:
    # Split by punctuation but keep the delimiters
    parts = re.split(r'([!?,.])', text)
    
    segments = []
    current_segment = ""
    
    for part in parts:
        # If the part is one of your punctuation marks
        if part in "!?,.":
            current_segment += part
            segments.append(current_segment.strip())
            current_segment = ""
        else:
            current_segment += part
            
    # Append any remaining text (e.g. " world" in "Hello, world")
    if current_segment.strip():
        segments.append(current_segment.strip())
        
    return segments

def get_id(audio_file_path: str) -> str:
    """
    a function to get the id from the audio file path
    """

    id = audio_file_path.split(".flac")[0]
    id = id.split("/")[-1]

    return id

def construct_spreedsheet_retaining_punctuation(
        json_data: list[dict]
) -> pd.DataFrame:
    """
    a function to form a spreedsheet for segmentic level segments for reconstructing the json file
    """

    # instantiate lists to store the data informations
    duration = []
    audio_file_path = []
    full_sentence = []
    segmentic_level_segments = []
    ids = []

    # loop through each json data
    for data in json_data:

        # get the id from the audio file path
        id = get_id(audio_file_path=data["audio_filepath"])

        # split the sentence wrt punctuation
        segments = split_sentence_retaining_punctuation(text=data["text"])

        # append the audio file path & full sentence to the list
        for _ in range(len(segments)):

            duration.append(data["duration"])
            audio_file_path.append(data["audio_filepath"])
            full_sentence.append(data["text"])
            ids.append(id)

        # add the segments to the list
        segmentic_level_segments.extend(segments)

    return pd.DataFrame({
        "duration": duration,
        "audio_file_path": audio_file_path,
        "full_sentence": full_sentence,
        "ids": ids,
        "segmentic_level_segments": segmentic_level_segments,
    })

def construct_spreedsheet_for_segmentic_level(
        json_data: list[dict]
) -> pd.DataFrame:
    """
    a function to form a spreedsheet for segmentic level segments for reconstructing the json file
    """

    # instantiate lists to store the data informations
    duration = []
    audio_file_path = []
    full_sentence = []
    segmentic_level_segments = []
    ids = []

    # loop through each json data
    for data in json_data:

        # get the id from the audio file path
        id = get_id(audio_file_path=data["audio_filepath"])

        # split the sentence wrt punctuation
        segments = split_sentence_wrt_punctuation(text=data["text"])

        # append the audio file path & full sentence to the list
        for _ in range(len(segments)):

            duration.append(data["duration"])
            audio_file_path.append(data["audio_filepath"])
            full_sentence.append(data["text"])
            ids.append(id)

        # add the segments to the list
        segmentic_level_segments.extend(segments)

    return pd.DataFrame({
        "duration": duration,
        "audio_file_path": audio_file_path,
        "full_sentence": full_sentence,
        "ids": ids,
        "segmentic_level_segments": segmentic_level_segments,
    })

def total_length_of_segments(segments: list[str]) -> int:
    total_length = 0
    for segment in segments:
        # Split by default (None) handles multiple spaces and tabs automatically
        list_segment = segment.strip().split() 
        total_length += len(list_segment)
    return total_length

def construct_segmentic_level_json(
        librispeech_pc_path: str,
        split: str,
        output_path: str
):
    """
    a function to construct the segmentic level segment in json format

    example of one json
    {
        "duration": 15.90
        "audio_file_path": "train-clean-100/2893/139310/2893-139310-0000.flac",
        "full_sentence": "Hello! How are you?",
        "chunks": [
            {
                "text": "Hello",
                "timestamp": [
                    0.1,
                    0.2
                ]
            },
            {
                "text": "How are you",
                "timestamp": [
                    0.4,
                    0.5
                ]
            }
        ]
    }
    """

    # instantiate a list to store the processed json data
    processed_json = []

    # read the json line file for librispeech pc
    librispeech_pc = read_json_line_file(json_file_path=librispeech_pc_path)

    # construct a spreadsheet for the segmentic level 
    librispeech_pc_df = construct_spreedsheet_for_segmentic_level(json_data=librispeech_pc)

    # download the forced alignment librispeech from hf
    librispeech_fa = download_forced_aligned_librispeech(split=split)

    # convert librispeech_fa to df
    librispeech_fa_df = librispeech_fa.select_columns(["id", "words"]).to_pandas()

    # get the unique ids from librispeech_pc
    unique_ids = librispeech_pc_df["ids"].unique().tolist()

    # get all the ids from librispeech_fa
    fa_ids = librispeech_fa_df["id"].unique().tolist()

    # loop through the id
    for id in unique_ids:

        if id in fa_ids:

            # get the audio_file_path, full_sentence, segmentic_level_segments, transcription from fa
            duration = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "duration"].tolist()[0]
            audio_file_path = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "audio_file_path"].tolist()[0]
            full_sentence = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "full_sentence"].tolist()[0]
            segmentic_level_segments = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "segmentic_level_segments"].tolist()
            word_level_forced_aligment = librispeech_fa_df.loc[librispeech_fa_df["id"] == id, "words"].tolist()[0]

            # instantiate the counter
            counter = 0
            
            # instantiate a list to store the chunks
            chunks = []

            # loop through the segmentic level segments
            if len(word_level_forced_aligment) == total_length_of_segments(segmentic_level_segments):

                for segment in segmentic_level_segments:

                    # get the length of the segment
                    words_in_segment = segment.strip().split()
                    len_segment = len(words_in_segment)

                    # Skip empty segments to prevent index errors
                    if len_segment == 0:
                        continue

                    # get the start time & end time
                    start_time = word_level_forced_aligment[counter]["start"]
                    end_time = word_level_forced_aligment[counter + len_segment - 1]["end"]

                    # update the counter
                    counter += len_segment

                    # construct the chunk data and append to the list
                    chunks.append(
                        {
                            "text": segment,
                            "timestamp": [
                                start_time,
                                end_time
                            ]
                        }
                    )

                # append the processed json to the list
                processed_json.append(
                    {
                        "duration": duration,
                        "audio_file_path": audio_file_path,
                        "full_sentence": full_sentence,
                        "chunks": chunks
                    }
                )

    # to write the output json file
    with open(output_path, "w", encoding="utf-8") as f:

        for datum in processed_json:

            f.write(json.dumps(datum) + "\n")

def construct_segment_level_json_w_punctuation(
    librispeech_pc_path: str,
    split: str,
    output_path: str
):
    
    # instantiate a list to store the processed json data
    processed_json = []

    # read the json line file for librispeech pc
    librispeech_pc = read_json_line_file(json_file_path=librispeech_pc_path)

    # construct a spreadsheet for the segmentic level 
    librispeech_pc_df = construct_spreedsheet_retaining_punctuation(json_data=librispeech_pc)

    # download the forced alignment librispeech from hf
    librispeech_fa = download_forced_aligned_librispeech(split=split)

    # convert librispeech_fa to df
    librispeech_fa_df = librispeech_fa.select_columns(["id", "words"]).to_pandas()

    # get the unique ids from librispeech_pc
    unique_ids = librispeech_pc_df["ids"].unique().tolist()

    # get all the ids from librispeech_fa
    fa_ids = librispeech_fa_df["id"].unique().tolist()

    # loop through the id
    for id in unique_ids:

        if id in fa_ids:

            # get the audio_file_path, full_sentence, segmentic_level_segments, transcription from fa
            duration = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "duration"].tolist()[0]
            audio_file_path = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "audio_file_path"].tolist()[0]
            full_sentence = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "full_sentence"].tolist()[0]
            segmentic_level_segments = librispeech_pc_df.loc[librispeech_pc_df["ids"] == id, "segmentic_level_segments"].tolist()
            word_level_forced_aligment = librispeech_fa_df.loc[librispeech_fa_df["id"] == id, "words"].tolist()[0]

            # instantiate the counter
            counter = 0
            
            # instantiate a list to store the chunks
            chunks = []

            # loop through the segmentic level segments
            if len(word_level_forced_aligment) == total_length_of_segments(segmentic_level_segments):

                for segment in segmentic_level_segments:

                    # get the length of the segment
                    words_in_segment = segment.strip().split()
                    len_segment = len(words_in_segment)

                    # Skip empty segments to prevent index errors
                    if len_segment == 0:
                        continue

                    # get the start time & end time
                    start_time = word_level_forced_aligment[counter]["start"]
                    end_time = word_level_forced_aligment[counter + len_segment - 1]["end"]

                    # update the counter
                    counter += len_segment

                    # construct the chunk data and append to the list
                    chunks.append(
                        {
                            "text": segment,
                            "timestamp": [
                                start_time,
                                end_time
                            ]
                        }
                    )

                # append the processed json to the list
                processed_json.append(
                    {
                        "duration": duration,
                        "audio_file_path": audio_file_path,
                        "full_sentence": full_sentence,
                        "chunks": chunks
                    }
                )

    # to write the output json file
    with open(output_path, "w", encoding="utf-8") as f:

        for datum in processed_json:

            f.write(json.dumps(datum) + "\n")   

def convert_format_to_training(
    path_of_processed_data: str,
    path_of_training_data: str
):
    """
    a function to convert the json line file to the format of training
    """

    # read the processed json data
    json_data = read_json_line_file(json_file_path=path_of_processed_data)

    # instantiate a list to store the data in training format
    training_formatted_json = []

    # define the system prompt, audio prompt
    system_prompt = {
        "role": "system",
        "content": "You are a speech recognition model. Transcribe the English audio into text with punctuation marks and capitalization."
    }

    audio_prompt = {
        "role": "user",
        "content": "<audio>"
    }

    # loop through the read json and convert the format
    for data in json_data:

        # retrive the duration, relative path of audio, ground truth & semantic level alignments
        duration = data["duration"]
        audio_relative_path = data["audio_file_path"]
        ground_truth = data["full_sentence"]
        chunks = data["chunks"]

        # construct the model response, audios path, & aligments
        #normalizer = str.maketrans("", "", string.punctuation)
        #normalized_ground_truth = ground_truth.translate(normalizer).lower()

        model_response = {
            "role": "assistant",
            "content": ground_truth
        }

        alignments = []
        for chunk in chunks:

            alignments.append({
                "text": chunk["text"],
                "start": round(chunk["timestamp"][0], 2),
                "end": round(chunk["timestamp"][1], 2)
            })

        # construct the message
        message = {
            "messages": [
                system_prompt,
                audio_prompt,
                model_response
            ],
            "audios": audio_relative_path,
            "duration": round(duration, 2),
            "alignments": alignments
        }

        # append the message to the training_formatted_json
        training_formatted_json.append(message)

    # write the output json file
    with open(path_of_training_data, "w", encoding="utf-8") as f:

        json.dump(training_formatted_json, f, indent=4, ensure_ascii=False)

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