
def visualize_configurations(yamlfile):
    for item in yamlfile.keys():
        print('\n--------------------------------------------------------')
        print('Configurations in phase {}'.format(item))
        print('--------------------------------------------------------\n')
        for details in yamlfile[item].keys():
            print('The key: {} is set to : {}'.format(details, yamlfile[item][details]))


def transfer_txt(opt):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(opt).items()):
            text += "|<<{}>>||:{}|  \n".format(attr, value)

        return text