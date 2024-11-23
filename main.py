from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import pandas as pd



class Assistant:
    def __init__(self,data=None):
        """
        Initialize the FAQ Assistant with DistilBERT model and tokenizer
        
        Args:
            data (not sure of the structure yet): not finalized yet
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.eval()


        self.questions = []
        self.answers = []
        self.embeddings = []
        

    def add_data(self,questions_and_answers):
        """
        Add FAQ questions and answers
        questions_and_answers: list of tuples [(question, answer), ...]
        """

        for question, answer in questions_and_answers:
            self.questions.append(question)
            self.answers.append(answer)

           


        self.embeddings = self._get_embeddings(self.questions)


    def _get_embeddings(self,texts):
        """Get embeddings for a list of texts"""

        embeddings = []
        # look this up still don't understand what toch.no_grad() really does
        with torch.no_grad():
            for text in texts:

                inputs = self.tokenizer(text,return_tensors="pt",padding=True, truncation=True)
                outputs = self.model(**inputs)

                embedding = outputs.last_hidden_state[:,0,:].numpy()
                embeddings.append(embedding[0])

            # print(embeddings)
            return np.array(embeddings)
    
    def answer_question(self,question, threshold=.7):
        """
        Answer a question by finding the most similar FAQ question
        Returns tuple (answer, confidence)
        """

        # get embedding for the question

        question_embedding = self._get_embeddings([question])

        # print(question_embedding)

        similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        index_of_best_match = np.argmax(similarities) # will return an index based on which question is asked, you can easily guess it if you pass the question manually, more on later
        confidence = similarities[index_of_best_match]


        # print(confidence) # expect it to be a value between 0 - 1
        # print(index_of_best_match)

        if confidence >=threshold:
            return self.answers[index_of_best_match], confidence
        else:
            return "I am as confused as you are right now!"

if __name__ == "__main__":
    assistant = Assistant()
    faqs = [
        ("How do I reset my password?", 
         "Click the 'Forgot Password' link on the login page and follow the email instructions."),
        ("Where can I find my account settings?", 
         "Account settings are in the top-right menu under your profile icon."),
        ("What payment methods do you accept?", 
         "We accept credit cards, PayPal, and bank transfers."),
    ]

    assistant.add_data(faqs)
        # Test some questions
    test_questions = [
        "How can I change my password?",
        "Where are the settings for my account?",
        "Do you take PayPal?",
        "What is the meaning of life?"  # This should return uncertain response
    ]


    answer, confidence = assistant.answer_question("Do you take PayPal?")
    print(answer,confidence)


    

  