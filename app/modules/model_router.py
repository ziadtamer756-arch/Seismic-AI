import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()



class ModelRouter:


    def __init__(self):

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )


        self.model = ChatGoogleGenerativeAI(

            model="gemini-3.6-flash",

            google_api_key=api_key,

            temperature=0

        )



    def run(self, task, data):


        prompt = f"""

You are a professional seismic researcher.

Analyze the following earthquake dataset.

Task:

{task}


Dataset:

{data}


Return a structured scientific report using:

# Summary

# Statistical Analysis

# Magnitude Distribution

# Depth Analysis

# Geographic Patterns

# Research Observations

# Conclusion

"""


        response = self.model.invoke(prompt)



        # Fix Gemini response format

        if hasattr(response, "content"):

            return response.content


        return str(response)