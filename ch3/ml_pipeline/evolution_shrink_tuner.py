import json
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner


class EvolutionShrinkTuner(EvolutionTuner):

    def __init__(self, optimize_mode = "maximize", population_size = 32):
        super().__init__(optimize_mode, population_size)

        # Shrunk Pipelines Registry
        self.registry = []

    def generate_parameters(self, *args, **kwargs):
        params = super().generate_parameters(*args, **kwargs)

        # If not `params` are not valid generate new ones
        while not self.is_valid(params):
            params = super().generate_parameters(*args, **kwargs)

        return params

    def is_valid(self, params):
        # All step names
        step_names = [v['_name'] for _, v in params.items() if v['_name'] != 'none']

        # No duplicates allowed
        if len(step_names) != len(set(step_names)):
            return False

        # `params` to canonical string
        canonical_form = 'X'
        for _, step_config in params.items():
            if step_config['_name'] == 'none':
                continue
            canonical_form += '--->' + json.dumps(step_config)

        # If `canonical_form` already tested
        if canonical_form in self.registry:
            return False

        self.registry.append(canonical_form)

        return True
