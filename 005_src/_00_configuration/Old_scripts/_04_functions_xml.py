from configuration._00_settings import *
from configuration._01_variables import *
from configuration._02_helper_functions import *

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
        # print (f"\ntime step {row_dict.values()} ------------------------------")
        for sub_e in e:
           # print ("")
            # print (sub_e.attrib)
            row_dict = {k:v for k,v in e.attrib.items()}
            row_dict_parameters = {k:v for k,v in sub_e.attrib.items()}
            row_dict.update(row_dict_parameters)
            # print (row_dict)
            list_of_dictionaries.append(row_dict)
    df = pd.DataFrame(list_of_dictionaries)
    return df