import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizerFast
from sklearn.model_selection import train_test_split
import pandas as pd

from config import question_labels, question_input_file, model_output_file, model_to_train

def train_model():

    loss = {
    "tf": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "mc": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "num": tf.keras.losses.MeanSquaredError()
}

    metrics = {
        "tf": tf.metrics.SparseCategoricalAccuracy('accuracy'),
        "mc": tf.metrics.SparseCategoricalAccuracy('accuracy'),
        "num": tf.metrics.RootMeanSquaredError()
    }


    transformer_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                    num_labels=question_labels[model_to_train]) 
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                                max_length=256,  
                                                pad_to_max_length=True) 

    data = pd.read_csv(question_input_file[model_to_train])
    data.dropna(inplace=True)

    X = data['question']
    y = data['label']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_encoded = dict(tokenizer(list(X_train.values),
                                    add_special_tokens=True,  
                                    max_length=256,  
                                    pad_to_max_length=True,  
                                    return_attention_mask=True))

    X_valid_encoded = dict(tokenizer(list(X_valid.values),
                                    add_special_tokens=True, 
                                    max_length=256,  
                                    pad_to_max_length=True,  
                                    return_attention_mask=True))

    train_data = tf.data.Dataset.from_tensor_slices((X_train_encoded, list(y_train.values)))
    valid_data = tf.data.Dataset.from_tensor_slices((X_valid_encoded, list(y_valid.values)))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-4)  

    transformer_model.compile(optimizer=optimizer,
                            loss=loss[model_to_train],
                            metrics=metrics[model_to_train])

    transformer_model.fit(train_data.shuffle(1000).batch(16),
                        epochs=1, batch_size=16,
                        validation_data=valid_data.batch(16))

    transformer_model.save_pretrained(model_output_file[model_to_train])
