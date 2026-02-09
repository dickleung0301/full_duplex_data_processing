import json
import torch
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Union, Optional

class TranslationDataset(Dataset):
    """
    a custom dataset to wrap the translation data
    """

    def __init__(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        utterance_ids: list[int],
        full_sentences: list[int]
    ):
        
        self.input_ids = torch.Tensor(input_ids).long()
        self.attention_mask = torch.Tensor(attention_mask).long()
        self.utterance_ids = torch.Tensor(utterance_ids).long()
        self.full_sentences = torch.Tensor(full_sentences).long()

    def __len__(self):

        return len(self.input_ids)
    
    def __getitem__(self, idx):

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "utterance_ids": self.utterance_ids[idx],
            "full_sentences": self.full_sentences[idx]
        }

def download_wmt_19() -> datasets.Dataset:
    """
    a function to download the validation set of wmt 19
    """

    return load_dataset(
        "wmt/wmt19",
        "zh-en",
        split="validation"
    )

def read_json_line_file(json_file_path: str) -> list[dict]:
    """
    a function to read the json line file
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

def extract_context_and_input_sentence(
    json_data: list[dict]
) -> Union[list[str], list[str], list[str], list[str], list[str]]:
    """
    a function to extract the previous context & also the input sentences
    """

    # instantiate lists to store the previous context & input sentences
    prev_context = []
    input_sentences = []
    full_sentences = []
    segments = []
    utterance_id = []

    # loop through the json_data to extract the pre context & input sentences
    for datum in json_data:

        context = ""
        id = datum["audio_file_path"].split("/")[-1].split(".flac")[0].strip()

        for i in range(len(datum["chunks"])):

            prev_context.append(context.strip())
            input_sentences.append(context + " " + datum["chunks"][i]["text"])
            segments.append(datum["chunks"][i]["text"])
            context = context + " " + datum["chunks"][i]["text"]
            full_sentences.append(datum["full_sentence"])
            utterance_id.append(id)

    return(prev_context, input_sentences, full_sentences, segments, utterance_id)


def apply_chat_template_with_in_context_example(
    num_in_context_example: int,
    input_sentences: list[str],
    tokenizer: AutoTokenizer,
    in_context_examples: Optional[dict] = None
) -> list[str]:
    """
    a function to construct prompt with in context examples
    """

    # instantiate the inference prompt
    inference_prompts = []

    # initialize the instruction
    instruction = "Translate the following English source text to Simplified Chinese:\n"

    # add the in context examples to the instruction
    if num_in_context_example > 0:

        for example in in_context_examples:

            instruction += "English: " + example["en"] + "\n" + "Simplified Chinese: " + example["zh"] + "\n"

    # construct the inference prompt
    for sentence in input_sentences:

        prompt = (instruction + "English: " + sentence + "\n" + "Simplified Chinese: ")
        inference_prompts.append([{
            "role": "user",
            "content": prompt
        }])

    # apply the chat template to the inference prompts
    chat_prompts = tokenizer.apply_chat_template(
        inference_prompts,
        tokenize=False,
        add_generation_prompt=True
        )

    return chat_prompts

def construct_translation_dataset(
    json_file_path: str,
    num_in_context_example: int,
    tokenizer: AutoTokenizer,
    max_length: int
) -> Dataset:
    """
    a function to construct the translation dataset
    """

    # read the json file
    json_data = read_json_line_file(json_file_path=json_file_path)

    # download wmt 19 dataset
    wmt19 = None

    if num_in_context_example > 0:

        wmt19 = download_wmt_19()
        wmt19 = wmt19["translation"][:num_in_context_example]

    # extract the previous context, input sentence, full sentence
    _, _, full_sentences, segments, utterance_ids = extract_context_and_input_sentence(json_data=json_data)

    # apply chat template to the input sentences
    chat_prompts = apply_chat_template_with_in_context_example(
        num_in_context_example=num_in_context_example,
        input_sentences=segments,
        tokenizer=tokenizer,
        in_context_examples=wmt19
    ) 

    # tokenize the input
    inputs = tokenizer(
        chat_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    tokenized_utterance_ids = tokenizer(
        utterance_ids,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    tokenized_full_sentences = tokenizer(
        full_sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )

    translation_data = TranslationDataset(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        utterance_ids=tokenized_utterance_ids["input_ids"],
        full_sentences=tokenized_full_sentences["input_ids"]
    )

    return translation_data

def constructing_prompt_for_streaming_mt(
    prev_generation: str,
    input_sentence: str,
    segment: str,
    tokenizer: AutoTokenizer
):
    """
    a function to construct a prompt for streaming mt
    """

    # define the system prompt
    system_prompt = "You are a helpful AI assistant in translation task."

    # construct the user prompt
    user_prompt = f"""Task: Translate the Current Segment defined below from English to Simplified Chineses.
                Context:
                Full Source Sentence: "{input_sentence}" (Use this for overall context and meaning).
                Frozen Translation (Already done): "{prev_generation}"
                Instructions:
                Translate the Current Segment so that it flows grammatically and logically from the Frozen Translation.
                Constraint 1: Do NOT re-translate, change, or include the "Frozen Translation" in your output.
                Constraint 2: Your output must contain ONLY the translation for the Current Segment.
                Constraint 3: Ensure gender/number agreement with the Frozen Translation.
                Current Segment to Translate: {segment}"""
    
    # construct the message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # apply the chat template to the messages
    chat_prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return chat_prompts

def trimmed_prompt(
    generation: str
) -> str:
    """
    a function to trim the generation prompt
    """

    return generation.split("assistant")[-1].strip()