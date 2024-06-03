import neptune

neptune_run = None

def neptune_init(name):
    """Init logging the run variables 

    Args:
        name (str): name of the python script
    """
    global neptune_run
    neptune_run = neptune.init_run(
        project="emaliemcmahon/SIEEG-analysis",
        name=name
    )


def neptune_params(params):
    """log the initial parameters for the run

    Args:
        params (dict): parameters used in running the code
    """
    global neptune_run
    for key, val in params.items():
        neptune_run[key] = val


def neptune_results(results):
    """save the results file

    Args:
        results (str): a path to an output results image
    """
    global neptune_run
    neptune_run["results"].upload(results)


def neptune_stop():
    """
    stop the neptune process
    """
    global neptune_run
    neptune_run.stop()
