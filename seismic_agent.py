from agent import SeismicAgent


def run_agent():

    agent = SeismicAgent()

    report = agent.analyze_waveform()

    return report