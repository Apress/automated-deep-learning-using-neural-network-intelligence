from ch3.ml_pipeline.operator import OperatorSpace


class SearchSpace:

    @classmethod
    def operator_search_space(cls, operator_type):
        """
        Search space for operator by `operator_type`
        """
        ss = []
        operators = []

        if operator_type == 'selector':
            # Selectors are not required in Pipeline
            ss.append({'_name': 'none'})
            operators = OperatorSpace.selectors
        elif operator_type == 'transformer':
            # Transformers are not required in Pipeline
            ss.append({'_name': 'none'})
            operators = OperatorSpace.transformers
        elif operator_type == 'classifier':
            operators = OperatorSpace.classifiers

        for o in operators:
            row = {'_name': o.name}
            for p_name, values in o.params.items():
                row[p_name] = {"_type": "choice", "_value": values}
            ss.append(row)

        return ss

    @classmethod
    def build(cls):
        return {
            "op_1": {
                "_type":  "choice",
                "_value": cls.operator_search_space('selector')
            },
            "op_2": {
                "_type":  "choice",
                "_value": cls.operator_search_space('transformer')
            },
            "op_3": {
                "_type":  "choice",
                "_value": cls.operator_search_space('transformer')
            },
            "op_4": {
                "_type":  "choice",
                "_value": cls.operator_search_space('transformer')
            },
            "op_5": {
                "_type":  "choice",
                "_value": cls.operator_search_space('classifier')
            }
        }


if __name__ == '__main__':
    search_space = SearchSpace.build()
    print(search_space)
