# Model parameters
PARAM = {
	'embed_dim' 	: 300,
	'hidden_dim' 	: 300,
	'n_layers' 		: 1,
	'dropout' 		: 0.5
}

# Data parameters
SEQ_LEN = 256

# Tokenizer parameters
VOCAB_SIZE = 8192
PADDING_SIZE = 256

# Define paths
PATH_DATA		= "./data/"
PATH_TOKENIZER 	= PATH_DATA + "serialized_tokenizer"
PATH_VOCAB 		= PATH_DATA + "vocab.json"
PATH_MERGES 	= PATH_DATA + "merges.txt"
PATH_TRAIN_TOK	= PATH_DATA + "tokenized_train"
PATH_VAL_TOK 	= PATH_DATA + "tokenized_val"
PATH_TEST_TOK 	= PATH_DATA + "tokenized_test"
PATH_DATA_FILES = [PATH_DATA + f"wiki.{split}.raw" for split in ["test", "train", "valid"]]
PATH_MODEL 		= "saved_model"