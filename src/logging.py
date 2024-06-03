import neptune
import subprocess

neptune_run = None

def get_short_git_hash():
    try:
        # Run the git command and capture the output
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Decode the output and strip any surrounding whitespace
        short_hash = result.stdout.decode('utf-8').strip()
        
        return short_hash
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to get the short git hash: {e.stderr.decode('utf-8').strip()}")
        return None


def neptune_init(name):
    """Init logging the run variables 

    Args:
        name (str): name of the python script
    """
    # Get git hash
    git_hash = get_short_git_hash()

    global neptune_run
    neptune_run = neptune.init_run(
                custom_run_id=f'{name}-{git_hash}',
                project="emaliemcmahon/SIEEG-analysis")


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
