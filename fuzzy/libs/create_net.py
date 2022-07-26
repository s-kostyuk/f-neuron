from fuzzy.nn import LeNetFuzzy, KerasNetFuzzy


def create_network(net_name: str, dataset_name: str, fuzzy_init: str = "Ramp"):
    if net_name == "LeNet":
        return LeNetFuzzy(flavor=dataset_name, fuzzy_fcn=False)

    if net_name == "LeNetFuzzy":
        return LeNetFuzzy(flavor=dataset_name, fuzzy_fcn=True, fuzzy_init=fuzzy_init)

    if net_name == "KerasNet":
        return KerasNetFuzzy(flavor=dataset_name, fuzzy_fcn=False)

    if net_name == "KerasNetFuzzy":
        return KerasNetFuzzy(flavor=dataset_name, fuzzy_fcn=True, fuzzy_init=fuzzy_init)

    raise NotImplemented("Networks other than LeNetFuzzy and KerasNetFuzzy are not supported.")

