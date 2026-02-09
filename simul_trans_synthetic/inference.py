import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from .open_source_model import(
    download_model_from_huggingface
)
from .load_dataset import(
    construct_translation_dataset,
    read_json_line_file,
    extract_context_and_input_sentence,
    constructing_prompt_for_streaming_mt,
    trimmed_prompt
)
from .closed_source_model import(
    deepl_api_endpoint,
    google_translate_api_endpoint
)
# from vllm import SamplingParams
# from .open_source_model import load_the_quantized_model
# from .data_processing import(
#     read_json_line_file,
#     construct_prompt_for_translation,
# )

# def offline_model_inference(
#     inference_prompts: list[str],
#     quantized_model_path: str
# ) -> list[str]:
#     """
#     a function to use vllm as the backend engine to do translation inference
#     """

#     # load the quantized LLM
#     llm = load_the_quantized_model(quantized_model_path=quantized_model_path)

#     # set up the generation config: greedy decoding
#     sampling_params = SamplingParams(
#         best_of=1,
#         temperature=0,
#         max_tokens=32,
#     )

#     # construct the conversation from the given list of prompts
#     conversations = []

#     for prompt in inference_prompts:

#         conversations.append([{
#             "role": "user",
#             "content": prompt
#         }])

#     # inference
#     outputs = llm.chat(conversations, sampling_params)

#     # extract the generation & return it
#     generations = []

#     for output in outputs:

#         generations.append(output.outputs[0].text)

#     return generations

# def offline_model_create_translation_synthetic_data(
#     json_file_path: str,
#     num_in_context_example: int,
#     quantized_model_path: str,
#     saving_path: str
# ):
#     """
#     a function to use offline model to create synthetic data
    
#     example of one json
#     {
#         "duration": 15.90
#         "audio_file_path": "train-clean-100/2893/139310/2893-139310-0000.flac",
#         "full_sentence": "Hello! How are you?",
#         "chunks": [
#             {
#                 "source_language": "Hello",
#                 "target_language": "",
#                 "previous_context": "",
#                 "timestamp": [
#                     0.1,
#                     0.2
#                 ]
#             },
#             {
#                 "source_language": "How are you",
#                 "target_language: "",
#                 "previous_context": "",
#                 "timestamp": [
#                     0.4,
#                     0.5
#                 ]
#             }
#         ]
#     }
#     """

#     # read the json line file
#     json_data = read_json_line_file(json_file_path=json_file_path)

#     # construct the prompt for MT task
#     inference_prompts, prev_context_list = construct_prompt_for_translation(
#         num_in_context_eample=num_in_context_example,
#         json_data=json_data
#     )

#     # do the offline inference
#     generations = offline_model_inference(
#         inference_prompts=inference_prompts,
#         quantized_model_path=quantized_model_path
#     )

#     # save the results
#     pd.DataFrame({
#         "prompts": inference_prompts,
#         "previous context": prev_context_list,
#         "translation": generations
#     }).to_csv(saving_path, index=False)

def mt_inference(
    model_path: str,
    quantization: bool,
    json_file_path: str,
    output_csv_path: str,
    num_in_context_example: int,
    max_length: int,
    batch_size: int
):
    """
    a function to carry out MT task inference
    """

    # download tokenizer & model from huggingface
    tokenizer, model = download_model_from_huggingface(
        model_path=model_path,
        quantization=quantization
    )

    # set the padding side
    tokenizer.padding_side = "left"

    # get the device of the model 
    device = model.device

    # load the inference dataset & wrap it with dataloader
    translation_dataset = construct_translation_dataset(
        json_file_path=json_file_path,
        num_in_context_example=num_in_context_example,
        tokenizer=tokenizer,
        max_length=max_length
    )

    translation_dataset_dataloader = DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # instantiate lists to store prompts, generations, prev_context, full_sentences
    prompts = []
    generations = []
    utterance_ids_list = []
    full_sentences = []

    for batch in tqdm(translation_dataset_dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        utterance_ids = batch["utterance_ids"]
        full_sentences_ids = batch["full_sentences"]

        # switch the dropout & layer norm of the model to eval mode
        model.eval()

        # turn off the gradient computation
        with torch.no_grad():

            # greedy decoding
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False
            )

        # decoding
        decoded_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_utterance_ids = tokenizer.batch_decode(utterance_ids, skip_special_tokens=True)
        decoded_full_sentences = tokenizer.batch_decode(full_sentences_ids, skip_special_tokens=True)

        # append the decoded string to the list
        prompts.extend(decoded_prompts)
        generations.extend(decoded_generations)
        utterance_ids_list.extend(decoded_utterance_ids)
        full_sentences.extend(decoded_full_sentences)

    # materialize the output
    df = pd.DataFrame({
        "prompts": prompts,
        "generations": generations,
        "utterance_ids": utterance_ids_list,
        "full_sentences": full_sentences
    })

    df.to_csv(output_csv_path, index=False)

def streaming_mt_inference(
    model_path: str,
    quantization: bool,
    json_file_path: str,
    output_csv_path: str,
    max_length: int,
):
    """
    a function to carry out streaming MT
    """

    # punctuation mapping
    punctuation_mapping = {
        
    }
    
    # download tokenizer & model from huggingface
    tokenizer, model = download_model_from_huggingface(
        model_path=model_path,
        quantization=quantization
    )

    # set the padding side
    tokenizer.padding_side = "left"

    # get the device of the model 
    device = model.device

    # read the json data
    json_data = read_json_line_file(json_file_path=json_file_path)

    # extract the previous context and input sentences
    prev_context, input_sentences, full_sentences, segments, utterance_id = extract_context_and_input_sentence(json_data=json_data)

    # instantiate a list to store the mt output & prompts
    mt_generation = []
    mt_segment = []
    prompts = []

    # carry out streaming MT
    for i in range(len(input_sentences)):

        # retrieve the past generation
        prev_generation = "" if prev_context[i] == "" else mt_generation[-1]

        # construct the chat prompt
        chat_prompt = constructing_prompt_for_streaming_mt(
            prev_generation=prev_generation,
            input_sentence=input_sentences[i],
            segment=segments[i],
            tokenizer=tokenizer
        )

        # tokenize the chat prompt
        inputs = tokenizer(
            chat_prompt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )

        # retrieve the attention mask and input ids
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0).to(device)

        # carry out the generation
        with torch.no_grad():

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False
            )

        # decode the generation
        decoded_generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # add the generation & prompt to the list
        mt_generation.append(prev_generation + trimmed_prompt(decoded_generation))
        mt_segment.append(trimmed_prompt(decoded_generation))
        prompts.append(chat_prompt)

    # materialize the output
    df = pd.DataFrame({
        "prev_context": prev_context,
        "full_sentences": full_sentences,
        "input_sentences": input_sentences,
        "segments": segments,
        "prompts": prompts,
        "generations": mt_generation,
        "mt_segments": mt_segment,
        "utterance_id": utterance_id
    })

    df.to_csv(output_csv_path, index=False)

def call_deepl_api(
    json_file_path: str,
    output_path: str
):
    """
    a function to call deepl api for mt task
    """

    # read the json file
    json_data = read_json_line_file(json_file_path=json_file_path)

    # extract the context and input sentences
    prev_context, input_sentences, full_sentences, _, _ = extract_context_and_input_sentence(json_data=json_data)

    # perform mt through deepl api
    mt_sentences = deepl_api_endpoint(input_sentences=input_sentences)

    # set up the spreadsheet & save it as csv
    df = pd.DataFrame({
        "previous_context": prev_context,
        "input_sentences": input_sentences,
        "generations": mt_sentences,
        "full_sentences": full_sentences
    })

    df.to_csv(output_path, index=False)

def call_google_translate_api(
    json_file_path: str,
    output_path: str
):
    """
    a function to call google translate api for mt task
    """

    # read the json file
    json_data = read_json_line_file(json_file_path=json_file_path)

    # extract the context and input sentences
    prev_context, input_sentences, full_sentences, _, _ = extract_context_and_input_sentence(json_data=json_data)

    # perform mt through deepl api
    mt_sentences = google_translate_api_endpoint(input_sentences=input_sentences)

    # set up the spreadsheet & save it as csv
    df = pd.DataFrame({
        "previous_context": prev_context,
        "input_sentences": input_sentences,
        "generations": mt_sentences,
        "full_sentences": full_sentences
    })

    df.to_csv(output_path, index=False)