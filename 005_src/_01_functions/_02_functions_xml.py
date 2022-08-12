from xml.parsers import expat
import xml.etree.ElementTree as ET

#https://stackoverflow.com/questions/28259301/how-to-convert-an-xml-file-to-nice-pandas-dataframe
class XmlParser(object):
    '''
    class used to retrive xml documents encoding
    '''

    def get_encoding(self, xml):
        self.__parse(xml)
        return self.encoding

    def __xml_decl_handler(self, version, encoding, standalone):
        self.encoding = encoding

    def __parse(self, xml):
        parser = expat.ParserCreate()
        parser.XmlDeclHandler = self.__xml_decl_handler
        parser.Parse(xml)

def xml_to_df(xml_data):
    """
    read the xml output into a df
    """
    etree = ET.parse(xml_data)
    list_of_dictionaries = []
    for e in etree.iter("timestep"):
        row_dict = {k:v for k,v in e.attrib.items()}
        for sub_e in e:
            row_dict = {k:v for k,v in e.attrib.items()}
            row_dict_parameters = {k:v for k,v in sub_e.attrib.items()}
            row_dict.update(row_dict_parameters)
            list_of_dictionaries.append(row_dict)
    df = pd.DataFrame(list_of_dictionaries)
    return df

def xml_to_dict(xml_data):
    """
    read the xml output into a dictionary
    """
    etree = ET.parse(xml_data)
    list_of_dictionaries = []
    for e in etree.iter("timestep"):
        row_dict = {k:v for k,v in e.attrib.items()}
        for sub_e in e:
            row_dict = {k:v for k,v in e.attrib.items()}
            row_dict_parameters = {k:v for k,v in sub_e.attrib.items()}
            row_dict.update(row_dict_parameters)
            list_of_dictionaries.append(row_dict)
     
    full_dict = {}
    for d in list_of_dictionaries:
        full_dict.update(d)
    return full_dict

def get_attr_details(xml_data, spacing_factor = 5, attr = "flow" ):
    """
    extract details of the attribute in attr
    """
    etree = ET.parse(xml_data)
    root = etree.getroot()
    all_rows = getDataRecursive(root)
    spacing = " "*spacing_factor
    return[ f"{spacing} {x}" for x in all_rows if attr in x]


def get_flow_details(xml_data, spacing_factor = 5):
    """
    extract details of the flow
    """
    etree = ET.parse(xml_data)
    root = etree.getroot()
    all_rows = getDataRecursive(root)
    spacing = " "*spacing_factor
    return[ f"{spacing} {x}" for x in all_rows if "flow" in x]

# https://stackoverflow.com/questions/51401826/parse-xml-in-python-without-manually-calling-attribute-tags-and-child-number
def getDataRecursive(element):
    """
    reads xml line by line into a data list
    """
    data = list()

    # get attributes of element, necessary for all elements
    ## ============================================================================================
    ## !!!! NOTE: important for reading to txt, dont leave spacing before the :, only after!!!!
    ## dont change this formatting <: > without changing the split_char in function "read_sim_data"
    ## ============================================================================================
    
    for key in element.attrib.keys():
        data.append(element.tag + '.' + key + ': ' + element.attrib.get(key))

    # only end-of-line elements have important text, at least in this example
    if len(element) == 0:
        if element.text is not None:
            data.append(element.tag + ': ' + element.text)

    # otherwise, go deeper and add to the current tag
    else:
        for el in element:
            within = getDataRecursive(el)

            for data_point in within:
                data.append(element.tag + '.' + data_point)

    return data

print (f"Functions xml import successful")