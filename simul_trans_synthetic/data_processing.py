# from .load_dataset import download_wmt_19

# def construct_prompt_for_translation(
#     num_in_context_eample: int,
#     json_data: list[dict]
# ) -> list[str]:
#     """
#     a function to construct the prompt for MT task
#     """

#     # instantiate a list to store the inference prompt
#     inference_prompts = []

#     # initialize the instruction
#     instruction = "Translate the following English source text to Simplified Chinese:\n"

#     # get the few shot examples
#     if num_in_context_eample > 0:

#         # download wmt 19 val set
#         wmt19 = download_wmt_19()

#         # extract the in-context-examples
#         translation_pairs = wmt19["translation"][:num_in_context_eample]

#         # add the in context examples to the instruction
#         for sample in translation_pairs:

#             instruction += "English: " + sample["en"] + "\n" + "Simplified Chinese: " + sample["zh"] + "\n"

#     # construct prompt for each segmentic level sentence
#     for datum in json_data:

#         # instantiate a str & a list to store previous context
#         prev_context = ""
#         prev_context_list = []

#         # extract the text chunks
#         chunks = datum["chunks"]

#         # loop through 
#         for i in range(len(chunks)):

#             # instantiate the delimiter for the segmentic level chunk
#             delimiter = ", " if i < (len(chunks) - 1) else "."

#             # construct the prompt
#             prompt = (instruction + "English: " 
#             + prev_context + chunks[i]["text"] + "\n"
#             + "Simplified Chinese: ")

#             # append the context to the list
#             prev_context_list.append(prev_context)

#             # update the previous context
#             prev_context = (prev_context + chunks[i]["text"] + delimiter)

#             # append the prompt to the list
#             inference_prompts.append(prompt)

#     return inference_prompts, prev_context_list