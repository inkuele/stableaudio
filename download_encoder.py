from transformers import T5Tokenizer, T5EncoderModel

#T5Tokenizer.from_pretrained("t5-small", cache_dir="./local_models/t5/")
#T5EncoderModel.from_pretrained("t5-small", cache_dir="./local_models/t5/")
T5Tokenizer.from_pretrained("t5-small")
T5EncoderModel.from_pretrained("t5-small")
