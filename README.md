# Tanjaro AI Guardian — Streamlit research prototype
Diya Shirwan and Dina Shirwan | Civil Engineering, Komar University of Science and Technology

## Before the showcase (Windows/macOS)
Install Python 3.11 or 3.12 from https://www.python.org/downloads/ (Windows: select Add Python to PATH). Extract this ZIP. Open a terminal inside the extracted folder.

Windows (Command Prompt):
```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```
If you installed Python 3.11, substitute -3.11.

macOS:
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```
Open http://localhost:8501 if the browser does not open automatically. Keep terminal running. Internet needed to install dependencies; local app works offline afterwards. Test before presenting. Do not upload private experimental data to public hosting.

## Online deployment (optional)
Create a GitHub repository, upload app.py, model.py, requirements.txt, .streamlit/config.toml. Sign in at https://share.streamlit.io/ and deploy app.py from the repository. No API key required by this code. Hosting not created by this package.

## Demonstration (2 minutes)
1. Open Monitoring dashboard, select Normal, then Possible blockage, then Poor filtration. Observe metrics, model classification and recommended inspection.
2. Select Mixed abnormality, then Sensor fault. Explain that invalid readings are screened before inference.
3. Record simulated sequence; inspect the trends. Export CSV.
4. Open Model evidence & limitations: compare Random Forest and threshold baseline on the same synthetic test split. The small difference does not establish real-world improvement.
5. Explain how calibrated ESP32 measurements can later be exported into the CSV schema. No direct hardware connection or plant actuation is implemented.

## Method and limits
The model is trained at startup on 2,000 synthetic samples and tested on 600 independently seeded synthetic samples from the same assumed distributions. Random Forest uses 80 trees. Input features: inlet and outlet turbidity, flow, derived reduction and flow ratio. The derived features contain no new sensor information. Four learned classes. Sensor fault and out-of-envelope/uncertain handling are deterministic safety checks, not learned diagnoses. Model vote share is not calibrated probability. Reference flow 2.5 L/min and all limits are demonstration assumptions, not regulatory limits. Low flow or high outlet turbidity has multiple possible causes. A zero flow falls outside model envelope and triggers review rather than a diagnosis. Inlet/outlet readings need residence-time alignment. The prototype cannot detect pathogens or heavy metals, certify discharge quality, or replace treatment infrastructure.

## Verification
Run `python test_model.py` inside the environment. TEST_RESULTS.txt records tests performed on the Python engine during creation. App syntax compiled successfully; Streamlit UI was not executed in the creation environment because Streamlit was unavailable and dependency download failed. Confirm browser operation on your laptop before the event.

## Files
app.py: Streamlit interface; model.py: training and diagnostics; test_model.py: reproducible checks; requirements.txt: dependencies; example_sensor_data.csv: simulated input; TEST_RESULTS.txt: observed test output. No external data or credentials required.
