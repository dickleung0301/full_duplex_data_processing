# from awq import AutoAWQForCausalLM
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
# from vllm import LLM
from typing import Union

# def create_a_quantized_llm(
#     model_path: str,
#     quantized_model_path: str
# ):
#     """
#     a function to create a quantized LLM
#     """

#     # initialize the quantization config
#     quant_config = {
#         "zero_point": True,
#         "q_group_size": 128,
#         "w_bit": 4,
#         "version": "GEMM"
#     }

#     # download the model
#     print(f"Loading {model_path}...")
#     model = AutoAWQForCausalLM.from_pretrained(
#         model_path,
#         **{"low_cpu_mem_usage": True, "use_cache": False}
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True
#     )

#     # quantize the model
#     model.quantize(tokenizer, quant_config=quant_config)

#     # save the quantized model
#     print(f"Saving to {quantized_model_path}...")
#     model.save_quantized(quantized_model_path)
#     tokenizer.save_pretrained(quantized_model_path)

# def load_the_quantized_model(
#     quantized_model_path: str
# ) -> LLM:
#     """
#     a function to load the quantized LLM
#     """

#     return LLM(
#         model=quantized_model_path,
#         quantization="awq"
#     )

def download_model_from_huggingface(
    model_path: str,
    quantization: bool
) -> Union[AutoTokenizer, AutoModelForCausalLM]:
    
    # set up the quantization config
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    # load the tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if quantization:

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quant_config
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )

    return (tokenizer, model)