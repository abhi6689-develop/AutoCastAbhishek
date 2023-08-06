from preprocess import process_dataset, create_csv_files
from model import train_model

if __name__ == "__main__":
    mc_qs, num_qs, tf_qs, qid_to_question = process_dataset()
    create_csv_files(mc_qs, num_qs, tf_qs)
    transformer_model = train_model()

