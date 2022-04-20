from nni.tuner import Tuner


class CustomTunerSample(Tuner):

    def __init__(self, some_arg) -> None:
        # YOUR CODE HERE #
        ...

    def update_search_space(self, search_space):
        """
        Tuners are advised to support updating search
        space at run-time. If a tuner can only set
        search space once before generating first
        hyper-parameters, it should explicitly document
        this behaviour. 'update_search_space' is called
        at the startup and when the search space is updated.
        """

        # YOUR CODE HERE #
        ...

    def generate_parameters(self, parameter_id, **kwargs):
        """
        This method will get called when the framework
        is about to launch a new trial. Each parameter_id
        should be linked to hyper-parameters returned by
        the Tuner. Returns hyper-parameters, a dict
        in most cases.
        """

        # YOUR CODE HERE #
        # Example: return {"dropout": 0.5, "act": "relu"}

        return {}

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        This method is invoked when a trial reports
        its final result. Should be implemented
        if Tuner assumes 'memory', i.e.,
        Tuner is tracking previous Trials
        """
        # YOUR CODE HERE #
