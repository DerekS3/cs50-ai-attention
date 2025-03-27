# CS50 AI Attention

Masked Language Model using BERT and the transformers Python library to predict missing words in sentences. BERT's attention heads are visualised, generating diagrams for each of its 144 attention heads. The project analyses these diagrams to interpret how BERT processes language.

## Contributions

`mask.py`:

`get_mask_token_index`: Takes a mask token ID and the tokenizer-generated inputs, returning the 0-indexed position of the mask token in the input sequence. Returns None if no mask token is present.

`get_color_for_attention_score`: Accepts an attention score (0 to 1) and returns an RGB tuple representing a shade of grey corresponding to that score. The colour is fully black for a score of 0 and fully white for a score of 1.

`visualise_attentions`: Generates attention visualisation diagrams for all attention heads and layers by indexing into the attentions tensor and passing layer and head numbers to the generate_diagram function.

### Testing

A test script (`test_mask.py`) has been developed to verify the correct operation of all listed functions.

### Technologies Used

- `Unittest`
- `TensorFlow`

### Usage

- main: `python3 mask.py`
- test: `python3 test_mask.py`