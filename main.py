import json
import argparse
from segment_level.data_processing import construct_segmentic_level_json, convert_format_to_training
# from simul_trans_synthetic.open_source_model import(
#     create_a_quantized_llm,
# )
from simul_trans_synthetic.inference import(
    # offline_model_create_translation_synthetic_data,
    mt_inference,
    call_deepl_api,
    call_google_translate_api,
    streaming_mt_inference
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentic_level", dest="segmentic_level", action="store_true", help="process the data in segmentic level")
    parser.add_argument("--training_format", dest="training_format", action="store_true", help="convert the json into training format")
    parser.add_argument("--pc_path", type=str, help="the path of the json file of librispeech_pc")
    parser.add_argument("--split", type=str, help="the split of librispeech")
    parser.add_argument("--output_path", type=str, help="the path of the output file")
    parser.add_argument("--processed_data_path", type=str, help="the path of the prcessed json file")
    parser.add_argument("--librispeech_dir", type=str, help="the dir of librispeech")
    parser.add_argument("--quantize", dest="quantize", action="store_true", help="to quantize the model")
    parser.add_argument("--inference", dest="inference", action="store_true", help="to infer the model")
    parser.add_argument("--streaming_inference", dest="streaming_inference", action="store_true", help="to streaming infer the model")
    parser.add_argument("--deepl_api", dest="deepl_api", action="store_true", help="to call the api from deepl")
    parser.add_argument("--google_trans_api", dest="google_trans_api", action="store_true", help="to call the api from google translate")
    parser.add_argument("--model", type=str, help="the model for inference")
    # parser.add_argument("--quantized_model_path", type=str, help="the path of the quantized model")
    parser.add_argument("--num_in_context_example", type=int, help="the number of in context examples for inference")
    parser.add_argument("--max_length", type=int, help="the maximum length in tokenization")
    parser.add_argument("--batch_size", type=int, help="the batch size in inference")

    args = parser.parse_args()
    segmentic_level = args.segmentic_level
    pc_path = args.pc_path
    split = args.split
    output_path = args.output_path
    training_format = args.training_format
    librispeech_dir = args.librispeech_dir
    processed_data_path = args.processed_data_path
    quantize = args.quantize
    inference = args.inference
    streaming_inference = args.streaming_inference
    deepl_api = args.deepl_api
    google_trans_api = args.google_trans_api
    model = args.model
    # quantized_model_path = args.quantized_model_path
    num_in_context_example = args.num_in_context_example
    max_length = args.max_length
    batch_size = args.batch_size

    # load the model mapping
    with open("mapping.json", "r") as f:

        model_mapping = json.load(f)["model"]

    if segmentic_level:

        construct_segmentic_level_json(
            librispeech_pc_path=pc_path,
            split=split,
            output_path=output_path
        )

    if training_format:

        convert_format_to_training(
            librispeech_dir=librispeech_dir,
            path_of_processed_data=processed_data_path,
            path_of_training_data=output_path,
        )

    # if quantize:

    #     create_a_quantized_llm(
    #         model_path=model_path,
    #         quantized_model_path=quantized_model_path
    #     )

    # if inference:

    #     offline_model_create_translation_synthetic_data(
    #         json_file_path=processed_data_path,
    #         num_in_context_example=num_in_context_example,
    #         quantized_model_path=quantized_model_path,
    #         saving_path=output_path
    #     )

    if inference:

        mt_inference(
            model_path=model_mapping[model],
            quantization=quantize,
            json_file_path=processed_data_path,
            output_csv_path=output_path,
            num_in_context_example=num_in_context_example,
            max_length=max_length,
            batch_size=batch_size
        )

    if streaming_inference:

        streaming_mt_inference(
            model_path=model_mapping[model],
            quantization=quantize,
            json_file_path=processed_data_path,
            output_csv_path=output_path,
            max_length=max_length
        )

    if deepl_api:

        call_deepl_api(
            json_file_path=processed_data_path,
            output_path=output_path
        )

    if google_trans_api:

        call_google_translate_api(
            json_file_path=processed_data_path,
            output_path=output_path           
        )