from test_cases import get_test_cases, get_toy_test_cases
import subprocess
import json


for case in get_test_cases():
    str_case = json.dumps(case)

    subprocess.run(["python", "configurable_CNN.py", str_case])
