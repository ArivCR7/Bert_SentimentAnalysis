import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 10
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = '../input/bert-base-uncased/'
MODEL_PATH = 'model.bin'
TRAINING_FILE = '../input/imdb.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case = True
)