from typing import List, Callable
import sys, os, time, io
import inspect
import html
import pandas as pd
import threading

class Logger(object):
    """ Experiment logger. With Singleton pattern, only one logger can exist in a process """
    _instance_lock = threading.Lock()
    _instance = None

    def __init__(self, name_str :str=None, output_dir: str='./log/', ):
        """init and reset
        NOTE: If 'open' method have been called before, 'close' method must be called!
        """
        self.terminal = sys.stdout
        self.log = None
        self.name_str = name_str
        self.output_dir = output_dir
        self._check_directory(output_dir)

        # other settings
        self.print_detail = True
        self.print_summary = False
        self.float_format = '%.4f'

    def __new__(cls, *args, **kwargs):
        """ 1> 一个对象的实例化过程是先执行类的__new__方法,如果我们没有写,默认会调用object的__new__方法,
        返回一个实例化对象,然后再调用__init__方法,对这个对象进行初始化,我们可以根据这个实现单例.
            2> 在一个类的__new__方法中先判断是不是存在实例,如果存在实例,就直接返回,如果不存在实例就创建.
        链接：https://www.jianshu.com/p/6a1690f0dd00"""
        if not cls._instance:
            with Logger._instance_lock:
                if not cls._instance:
                    Logger._instance = super().__new__(cls)

        return Logger._instance

    @classmethod
    def get_instance(cls, *args, **kwargs) -> 'Logger':
        """ return Singleton instance """
        if not cls._instance:
            raise ValueError('You need init this class first!')
        return cls._instance

    @staticmethod
    def _check_directory(dir_path: str):
        """check and make directory"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def set_print(self, print_detail: bool=True, print_summary: bool=False):
        self.print_detail = print_detail
        self.print_summary = print_summary

    def set_float_format(self, float_format: str='%.4f'):
        self.float_format = float_format

    def open(self, file_name = None):
        if file_name is None:
            file_name = self._gen_detail_file_name()
        self.log = open(file_name, 'wt', newline='\n')
    
    def close(self):
        self.log.close()

    def _gen_detail_file_name(self):
        """Deatil log filename formart: ./log/<Year-Month-Day>_detail_<name_str>_<Pid>.log"""
        date_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        pid = os.getpid()
        file_name = '%s/%s_detail_%s_%d.log' % (self.output_dir, date_str, self.name_str, pid)
        return file_name

    def _gen_summary_filename(self):
        """Summary filename formart: ./log/<Year-Month-Day>_<name_str>.log"""
        date_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        file_name = self.output_dir + date_str + '_' + self.name_str + '.log'
        return file_name

    def log_timestamp(self):
        """ Write time string to detailed log """
        headstr = '\n' + '====' * 20 + '\n'
        timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        message = '%sTime: %s    ProcessID: %d\n' % (headstr, timestr, os.getpid())
        self.log_detail(message)

    def log_detail(self, message: str):
        """ Write detailed log. If the log file exists, append output to the file.
        """
        message += '\n'
        # print to terminal
        if self.print_detail:
            self.terminal.write(message)
            self.terminal.flush()
        # print to log file
        self.log.write(message)
        self.log.flush()

    def log_summary(self, report_dict):
        """ Save experiment summary, include time and result.
        Summary filename formart: ./log/<Year-Month-Day>_<name_str>.log
        If the log file exists, append output to the file.
        Param:
            report_dict: dict, key is a str, value must be convertible into a string or be a DataFrame.
        """
        file_name = self._gen_summary_filename()
        timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        with open(file_name, mode='a') as f:
            # write head
            f.write('\n' + '====' * 20 + '\n')
            f.write('Time: %s    PID:%d\n' % (timestr, os.getpid()))
            # write key - value
            for key in report_dict:
                f.write('%s:' % key)
                value = report_dict[key]
                if type(value) == pd.DataFrame:
                    f.write('\n')
                    value.to_csv(f, sep='\t', float_format=self.float_format)
                else:
                    f.write(str(value))
                f.write('\n')
            f.write('\n')

    @staticmethod
    def dataframe2TSVstr(df: pd.DataFrame, float_format: str='%.4f') -> str:
        """ Convert pd.DataFrame data to TSV-style multi-line string
        """
        output = io.StringIO(newline='\n')
        # output.write('First line.\n')
        df.to_csv(output, sep='\t', float_format=float_format)

        # Retrieve file contents
        contents = output.getvalue()
        # Close object and discard memory buffer
        output.close()
        return contents

    @staticmethod
    def func2str(fun: Callable, replace_escape_char=True) -> str:
        """
        Hash functions

        :param fun: function to hash
        :return: hash of the function
        """
        try:
            h = inspect.getsource(fun)
            if replace_escape_char:
                # h = h.replace('\\n', '\n').replace(r"\'", r"'").replace(r'\"', r'"')
                h = html.unescape(h)
        except IOError:
            h = "nocode"
        return h

if __name__ == '__main__':
    print('Start test this class! Pid %d' % os.getpid())
    exp_logger = Logger(name_str='test', output_dir='./test/')
    Logger.get_instance().open()
    Logger.get_instance().log_timestamp()
    Logger.get_instance().log_detail('hello, this is a test')
    Logger.get_instance().log_detail('This is the second line')
    Logger.get_instance().log_detail('You may want to delete this directory.')
    Logger.get_instance().log_detail('@function:\n' + Logger.func2str(Logger.dataframe2TSVstr))
    df = pd.DataFrame({"col1":['A', 'B', 'C'], "col2":[1, 2, 3]})
    Logger.get_instance().log_detail('@df:\n' + Logger.dataframe2TSVstr(df))
    Logger.get_instance().close()
