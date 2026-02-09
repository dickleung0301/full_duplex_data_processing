import os
import deepl
from dotenv import load_dotenv
# from googletrans import Translator
from deep_translator import GoogleTranslator

def deepl_api_endpoint(
    input_sentences: list[str]
) -> list[str]:
    """
    an API endpoint of DeepL
    """

    # instantiate a list to store the machine translated sentences
    mt_sentences = []

    # load the environmental variable
    load_dotenv()

    # load the auth key for deepl
    deepl_auth_key = os.getenv("DEEPL_AUTH_KEY")

    # set up the http connection
    deepl_client = deepl.DeepLClient(auth_key=deepl_auth_key)

    # request client for mt task
    results = deepl_client.translate_text(
        input_sentences,
        target_lang="ZH-HANS"
    )

    # to extract the results
    for result in results:

        mt_sentences.append(result.text)

    return mt_sentences

def google_translate_api_endpoint(
    input_sentences: list[str]
) -> list[str]:
    """
    an API endpoint of google translate
    """

    # set up the http connection
    translator = GoogleTranslator(source="en", target='zh-CN')

    # call google translate api
    translations = translator.translate_batch(input_sentences)

    return translations