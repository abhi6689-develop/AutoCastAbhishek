
# Autocast Forecasting with BERT

A forecasting tool that leverages the BERT model to provide nuanced and context-aware predictions.

## Table of Contents
- [File Descriptions](#file-descriptions)
- [Setup and Usage](#setup-and-usage)
- [Further Files](#further-files)

## File Descriptions

1. `config.py` - Contains configuration settings for the project. This includes mappings of question types to their respective labels, input data file paths, and the paths for the saved models.
  
2. `main.py` - The primary script to preprocess the dataset, create CSV files, and initiate the model training process.
  
3. `model.py` - Defines the function `train_model()` that preprocesses the dataset, tokenizes it, splits it into training and validation sets, and then trains the BERT model based on the configurations in `config.py`.

4. `preprocess.py` - Contains various utility functions to preprocess the data:
    - `convert_questions_to_fid_format()` - Converts questions to FID format and saves them to JSON.
    - `load_questions()` - Loads question data and its negated version.
    - `separate_questions_by_type()` - Separates questions based on their type (multiple choice, numeric, true/false).
    - `separate_questions_by_status()` - Separates questions based on their status and specified time.
    - `create_question_dict()` - Returns a dictionary mapping question IDs to their details.
    - `process_dataset()` - Processes the dataset and returns questions sorted by type and an ID to question dictionary.
    - `create_csv_files()` - Creates CSV files for different types of questions.

## Setup and Usage

1. Ensure you have all the required libraries installed. This project primarily relies on `tensorflow`, `transformers`, `datasets`, and `sklearn`.

2. Set the desired model you want to train in `config.py` by modifying the `model_to_train` variable. It can be set to `"tf"`, `"mc"`, or `"num"`.

3. Run the main script:
```bash
python main.py
```

This will initiate the data preprocessing, CSV file creation, and the model training processes. Once the model has been trained, it will be saved to the designated output file path as set in `config.py`.
Let's continue the `README.md` by adding details about the `submission.py` file:


5. `submission.py` - A script to generate predictions for test questions and save the output in a serialized format:
    - **Tokenizer Initialization**: Uses the `BertTokenizerFast` to initialize a tokenizer that is specific to BERT. This tokenizer handles padding and truncating of sequences.
    - **Model Loading**: Loads three separate BERT models trained for multiple choice, true/false, and numeric type questions respectively.
    - `format_output(output, size)` - Modifies the output of the model based on the provided size, normalizes the values, and ensures the sum of probabilities is 1.
    - `get_answer(question)` - Takes a question as input and returns the model's prediction based on the question type.
    - Main Process: Iterates through each test question, gets its prediction, and appends it to a list. Finally, saves the list of predictions as a pickle file in a directory named 'submission'.

## Usage for Generating Submissions

To generate predictions for the test set, simply run:

```bash
python submission.py
```

Upon execution, the script will load the pretrained models, tokenize the test questions, obtain model predictions, and save the predictions in a pickle file inside the 'submission' directory.

---
