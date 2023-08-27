import sys
import logging
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    fileName=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name[{0}] line number [{1}] error message [{2}]".format(fileName,exc_tb.tb_lineno,str(error))
    return error_message
class CustopException(Exception):
    def __init__(self,err_message,error_details:sys):
        super().__init__(err_message)
        self.errorMessage=error_message_detail(err_message,error_details)

    def __str__(self) :
        return self.errorMessage
 