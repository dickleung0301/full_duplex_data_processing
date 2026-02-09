from transformers import AutoTokenizer
from load_dataset import constructing_prompt_for_streaming_mt
import torch
from load_dataset import TranslationDataset
from torch.utils.data import DataLoader, DataLoader
from tqdm import tqdm

# messages = [
#     [{"role": "user", "content": "Translate the following English source text to Portuguese (Portugal):\nEnglish: Hello world!\nPortuguese (Portugal): "}],
#     [{"role": "user", "content": "Translate the following English source text to Portuguese (Portugal):\nEnglish: Hello world!\nPortuguese (Portugal): "}],
#     ]
tokenizer = AutoTokenizer.from_pretrained("Unbabel/Tower-Plus-72B")
# prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(prompts)

# dataset = construct_translation_dataset(
#     json_file_path = "../processed_data/simul_tran_dummy_data.json",
#     num_in_context_example = 3,
#     tokenizer = tokenizer,
#     max_length = 256
# )

# print(dataset)
# print(dataset["input_ids"])
# print(type(dataset["input_ids"]))
# print(dataset["attention_mask"])
# print(type(dataset["attention_mask"]))
# print(dataset["prev_context"])
# print(type(dataset["prev_context"]))
# print(dataset["full_sentences"])
# print(type(dataset["full_sentences"]))

# input_ids = torch.Tensor([[1,0], [0,1]])
# attention_mask = torch.Tensor([[1,1], [1,1]])
# prev_context = torch.Tensor([[0,0], [1,1]])
# full_sentences = torch.Tensor([[0,0], [1,1]])

# translation_data = TranslationDataset(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     prev_context=prev_context,
#     full_sentences=full_sentences
# )

# translation_dataloader = DataLoader(translation_data, batch_size=2)

# for batch in tqdm(translation_dataloader):

#     print(batch["input_ids"])
#     print(batch["attention_mask"])
#     print(batch["prev_context"])
#     print(batch["full_sentences"])

chat_prompt = constructing_prompt_for_streaming_mt(
    prev_generation="",
    input_sentence="Hi",
    segment="Hi",
    tokenizer=tokenizer
)

print(type(chat_prompt))