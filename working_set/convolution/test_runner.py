from test_cases import get_test_cases, get_toy_test_cases, get_robust_easy, get_offenders
import subprocess
import json


# for case in get_test_cases():
for case in get_toy_test_cases():
# for case in get_offenders():
    str_case = json.dumps(case)

    subprocess.run(["python", "configurable_CNN.py", str_case])
