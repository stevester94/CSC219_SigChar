import json


with open("test_results.json", "r") as results:
    results_str = results.read()

results = json.loads(results_str)

for r in [r for r in results if (float(r["time"]) > 2000 and len(r["target"][0]) == 24 and len(r["target"][1]) == 6) or ("label" in r.keys() and "SEPARATOR" in r["label"])]:
    print("===========================================================================================================")
    try:
        print("label: %s, accuracy: %s, num_train_epochs:%s, len_mods: %s, len_snr: %s, time: %s" % (r["label"], r["accuracy"], r["num_train_epochs"], len(r["target"][0]), len(r["target"][1]), r["time"]))

    except:
        print("label: %s, accuracy: %s, num_train_epochs:%s, len_mods: %s, len_snr: %s, time: %s" % ("NONE", r["accuracy"], r["num_train_epochs"], len(r["target"][0]), len(r["target"][1]), r["time"]))

    # print("    network_conv_settings: %s" % r["network_conv_settings"])
    # print("    network_fc_settings: %s" % r["network_fc_settings"])
