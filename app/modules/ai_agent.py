from langchain_google_genai import ChatGoogleGenerativeAI
from .model_manager import ModelManager



class SeismicAI:


    def __init__(self):

        manager = ModelManager()


        self.models = (
            manager.available_models()
        )


        self.gemini = None


        if "Gemini" in self.models:

            self.gemini = ChatGoogleGenerativeAI(

                model="gemini-3.6-flash",

                temperature=0

            )



    def analyze(self, data):


        prompt = f"""

You are a seismic research scientist.

Analyze earthquake dataset:

{data}


Provide:

1. Activity level
2. Geographic patterns
3. Depth interpretation
4. Possible tectonic explanation
5. Research recommendations

"""


        if self.gemini:


            response = self.gemini.invoke(
                prompt
            )


            return response.content



        return "No AI model available"