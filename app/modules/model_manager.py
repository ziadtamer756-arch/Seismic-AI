import os

from dotenv import load_dotenv


load_dotenv()



class ModelManager:


    def __init__(self):

        self.gemini_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        self.openai_key = os.getenv(
            "OPENAI_API_KEY"
        )



    def available_models(self):

        models = []


        if self.gemini_key:
            models.append(
                "Gemini"
            )


        if self.openai_key:
            models.append(
                "OpenAI"
            )


        models.append(
            "Local ML"
        )


        return models