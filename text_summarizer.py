import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from rouge import Rouge
import warnings

warnings.filterwarnings("ignore")


class Text_Summarizer():

    def __init__(self, data):
        self.max_length_input = 1024
        self.max_length_output = 150
        self.num_beams = 8

        # Model names and their corresponding text summarization functions
        models = {'T5': self.summarize_text_t5,
                  'BART': self.summarize_text_bart,
                  'Pegasus': self.summarize_text_pegasus
                  }

        # Summarize texts using the T5, BART and Pegasus
        model_names = list(models.keys())
        generated_summaries = {model_name: [] for model_name in model_names}

        for model_name, model_function in models.items():
            data[model_name + '_summaries'] = data['Articles'].apply(lambda x: model_function(x))
            generated_summaries[model_name] = data[model_name + '_summaries'].tolist()

        references = data['Summaries'].tolist()
        rouge_scores = self.calculate_rouge_scores(model_names, generated_summaries.values(), references)

        print("ROUGE Scores:")
        for model_name, scores in rouge_scores.items():
            print(f"{model_name}:")
            for metric, score in scores.items():
                print(f"\t{metric}: {score}")

        print("\nArticles with Generated Summaries:")
        for model_name in models.keys():
            print(data[['Articles', model_name + '_summaries']].head())
            print()


    # Text summarization function using T5
    def summarize_text_t5(self, text):
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base')

        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=self.max_length_input, truncation=True)
        outputs = model.generate(inputs, max_length=self.max_length_output, num_beams=self.num_beams, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary


    # Text summarization function using BART
    def summarize_text_bart(self, text):
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=self.max_length_input, truncation=True)
        summary_ids = model.generate(input_ids, max_length=self.max_length_output, num_beams=self.num_beams, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary


    # Text summarization function using Pegasus
    def summarize_text_pegasus(self, text):
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
        model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')

        inputs = tokenizer([text], max_length=self.max_length_input, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=self.max_length_output, num_beams=self.num_beams, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary


    # Function for calculating ROUGE scores
    def calculate_rouge_scores(self, model_names, generated_summaries, references):
        rouge_scores = {}
        rouge = Rouge()

        for model_name, generated_summary in zip(model_names, generated_summaries):
            scores = rouge.get_scores(generated_summary, references, avg=True)
            rouge_scores[model_name] = scores

        return rouge_scores


def main():
    # Load the BBC news data (taken from: https://www.kaggle.com/datasets/pariza/bbc-news-summary)
    df = pd.read_csv('bbc_news.csv')

    # Considering 20 rows from the top for text summarization
    data = df.head(20)
    Text_Summarizer(data)


if __name__ == "__main__":
    main()
