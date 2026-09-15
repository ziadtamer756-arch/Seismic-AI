import os
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
from obspy import read
from google import genai


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise Exception("GOOGLE_API_KEY missing")


client = genai.Client(
    api_key=API_KEY
)


class SeismicAgent:


    def analyze_waveform(self, file_path=None):

        print("Loading seismic data...")


        if file_path and os.path.exists(file_path):

            print("Reading seismic file...")

            stream = read(file_path)

            trace = stream[0]

            data = trace.data


            info = {
                "station": trace.stats.station,
                "network": trace.stats.network,
                "sampling_rate": float(trace.stats.sampling_rate),
                "duration": float(
                    trace.stats.endtime -
                    trace.stats.starttime
                ),
                "max_amplitude": float(np.max(data)),
                "min_amplitude": float(np.min(data))
            }


        else:

            print("No seismic file found. Creating test waveform...")


            data = np.random.normal(
                0,
                1,
                5000
            )


            info = {
                "station": "TEST",
                "network": "LOCAL",
                "sampling_rate": 100,
                "duration": 50,
                "max_amplitude": float(np.max(data)),
                "min_amplitude": float(np.min(data))
            }



        prompt = f"""

You are a professional seismic engineer.

Analyze this seismic waveform data:

{info}


Create a professional seismic report:

1. Signal quality
2. Noise level
3. Possible seismic activity
4. Risk assessment
5. Engineering recommendations

Write a detailed technical report.

"""


        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            return response.text


        except Exception as e:

            return f"""

Gemini Error:

{str(e)}


Local seismic data:

{info}

"""



if __name__ == "__main__":


    agent = SeismicAgent()


    report = agent.analyze_waveform()


    print("\n========== REPORT ==========\n")

    print(report)



    os.makedirs(
        "reports",
        exist_ok=True
    )


    filename = (
        "reports/report_"
        +
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        +
        ".txt"
    )


    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)



    print("\nReport saved:")
    print(filename)