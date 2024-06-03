import neptune
import git


neptune_run = None

def neptune_init(name):
    """Init logging the run variables 

    Args:
        name (str): name of the python script
    """
    # Get git hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    global neptune_run
    neptune_run = neptune.init_run(
        project="emaliemcmahon/SIEEG-analysis",
        custom_run_id=f'{name}-{sha}',
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


def neptune_results(figure):
    """save the figure with neptune

    Args:
        figure (str or matplotlib.figure.Figure): a path to an output results image
            or a figure object to be uploaded
    """
    global neptune_run
    neptune_run["results"].upload(figure)


def neptune_stop():
    """
    stop the neptune process
    """
    global neptune_run
    neptune_run.stop()
