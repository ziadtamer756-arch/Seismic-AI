from earthquake_data import get_recent_earthquakes
from datetime import datetime
from dotenv import load_dotenv
from google import genai
import os


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def analyze_earthquakes(events):

    prompt = f"""

You are a seismic research assistant.

Analyze these recent earthquake events:

{events}


Create a scientific summary including:

1. Number of events
2. Highest magnitude event
3. Locations
4. Depth analysis
5. Risk interpretation
6. Research observations


Write a professional earthquake monitoring report.

"""


    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    return response.text



if __name__ == "__main__":


    print("Fetching earthquake data...")


    events = get_recent_earthquakes()


    report = analyze_earthquakes(events)


    print("\n========== EARTHQUAKE REPORT ==========\n")

    print(report)


    with open(
        "reports/daily_earthquake_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)


    print("\nReport saved.")