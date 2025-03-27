import unittest
import numpy as np
import os

from unittest.mock import patch
from mask import *


class TestGetMaskTokenIndex(unittest.TestCase):
    def setUp(self):
        None
        
    def test_get_mask_token_index_position_5(self):
        text = "The cat could smell a [MASK] from the table."
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(text, return_tensors="tf")
        mask_token_index = get_mask_token_index(
            tokenizer.mask_token_id, inputs
        )
        self.assertEqual(mask_token_index, 5)

    def test_get_mask_token_index_no_mask(self):
        text = "The cat could smell a from the table."
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(text, return_tensors="tf")
        mask_token_index = get_mask_token_index(
            tokenizer.mask_token_id, inputs
        )
        self.assertEqual(mask_token_index, None)


class TestColorForAttentionScore(unittest.TestCase):
    def setUp(self):
        None
        
    def test_color_for_attention_score_black(self):
        attention_score = 0
        actual_colour = get_color_for_attention_score(attention_score)
        expected_colour = (0, 0, 0)
        self.assertEqual(actual_colour, expected_colour)

    def test_color_for_attention_score_grey(self):
        attention_score = 0.5
        actual_colour = get_color_for_attention_score(attention_score)
        expected_colour = (127, 127, 127)
        self.assertEqual(actual_colour, expected_colour)


class TestVisualizeAttentions(unittest.TestCase):    
    def setUp(self):
        None
    
    def tearDown(self):
        for f in os.listdir('imgs'):
            if f.startswith('Attention_') and f.endswith('.png'):
                os.remove(f)
    
    @patch('mask.generate_diagram')
    def test_visualize_attentions_call_count(self, mock_generate_diagram):
        text = "The cat could smell a from the table."
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(text, return_tensors="tf")
        model = TFBertForMaskedLM.from_pretrained(MODEL)
        result = model(**inputs, output_attentions=True)

        visualize_attentions(inputs.tokens(), result.attentions)

        self.assertEqual(mock_generate_diagram.call_count, 144)


if __name__ == '__main__':
    unittest.main()