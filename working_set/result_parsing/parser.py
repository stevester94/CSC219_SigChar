import json

RESULTS_PATH = "../convolution/test_results.json"

with open(RESULTS_PATH, 'r') as content_file:
    results_str = content_file.read()

    results = json.loads(results_str)

    for r in results:
        try:
            print(r["label"])
        except:
            print("No label")
        print("Num epochs: %d" % r["num_train_epochs"])
        print("Accuracy: %s" % str(r["accuracy"]))
        print("Time (seconds): %f" % float(r["time"]))
        print("--------------------------")
