import nni


def main(args):
    result = args['x'] * 2
    nni.report_final_result(result)


if __name__ == '__main__':
    main(nni.get_next_parameter())
