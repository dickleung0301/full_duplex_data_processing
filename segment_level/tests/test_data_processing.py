import unittest
from datasets import Dataset
from imports import (
    read_json_line_file,
    split_sentence_wrt_punctuation,
    construct_spreedsheet_for_segmentic_level,
    get_id,
    download_forced_aligned_librispeech
)

class TestDataProcessing(unittest.TestCase):
    """
    test case for data processing module
    """

    def setUp(self):
        
        print("start testing...")

    def tearDown(self):
        
        print("end testing...")

    def test_read_json_line_file(self):
        """
        a test case for reading the json line file
        ------------------------
        expected: 
        json_data = read_json_line_file(json_file_path="../../manifests/train-clean-100.json")
        print(type(json_data)) -> list
        print(type(json_data[0])) -> dict
        print(len(json_data)) -> 26041
        """

        json_data = read_json_line_file(json_file_path="../../manifests/train-clean-100.json")

        self.assertIsInstance(obj=json_data, cls=list)
        self.assertIsInstance(obj=json_data[0], cls=dict)
        self.assertEqual(len(json_data), 26041)

    def test_split_sentence_wrt_punctuation(self):
        """
        a test case for split_sentence_wrt_punctuation
        ------------------------
        expected: 
        segments = split_sentence_wrt_punctuation(text="Hello!!!!!! How are you?  ")

        print(len(segments)) -> 2
        print(segments) -> "Hello", "How are you"
        """

        segments = split_sentence_wrt_punctuation(text="Hello!!!!!! How are you?  ")
        expected_segements = ["Hello", "How are you"]

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments, expected_segements)

    def test_construct_spreedsheet_for_segmentic_level(self):
        """
        a test case for construct_spreedsheet_for_segmentic_level
        ------------------------
        expected: 
        dummy_json = {"duration": 14.91, 
        "audio_filepath": "train-clean-100/2893/139310/2893-139310-0000.flac", 
        "text": "Hello!!!!!! How are you?  ."}

        segments = construct_spreedsheet_for_segmentic_level(json_data=dummy_json)

        print(len(spreedsheet)) -> 2
        spreedsheet["audio_filepath"][0] == segments["audio_filepath"][1]
        spreedsheet["full_sentence"][0] == segments["full_sentence"][1]
        spreedsheet["segmentic_level_segments"][0] == "Hello"
        spreedsheet["segmentic_level_segments"][1] == "How are you"
        """

        dummy_json = [{"duration": 14.91, 
        "audio_filepath": "train-clean-100/2893/139310/2893-139310-0000.flac", 
        "text": "Hello!!!!!! How are you?  ."}]

        spreedsheet = construct_spreedsheet_for_segmentic_level(json_data=dummy_json)

        self.assertEqual(len(spreedsheet), 2)
        self.assertEqual(spreedsheet.loc[0, "audio_file_path"], spreedsheet.loc[1, "audio_file_path"])
        self.assertEqual(spreedsheet.loc[0, "full_sentence"], spreedsheet.loc[1, "full_sentence"])
        self.assertEqual(spreedsheet.loc[0, "segmentic_level_segments"], "Hello")
        self.assertEqual(spreedsheet.loc[1, "segmentic_level_segments"], "How are you")

    def test_get_id(self):

        dummy_file_path = "train-clean-100/2893/139310/2893-139310-0000.flac"
        id = get_id(audio_file_path=dummy_file_path)

        self.assertEqual(id, "2893-139310-0000")

    def test_download_forced_aligned_librispeech(self):

        data = download_forced_aligned_librispeech(split="dev_clean")
        self.assertIsInstance(obj=data, cls=Dataset)

if __name__ == "__main__":

    unittest.main()