#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Chunguang Zhang
@contact: cgzhang6436@.com
@github: https://github.com/Franciszz
@file: logger.py
@time: 2019-11-17 13:56
@desc:
"""

import os
import logging
import time

from contextlib import contextmanager


class Logger:

    def __init__(self, filename="logs", dir_name="logs",
                 threading=False, multiprocessing=False,
                 console_mode=True, console_level=logging.DEBUG,
                 file_mode=True, file_level=logging.DEBUG, covered=False):
        """
        :param filename: filename of logs.
        :param dir_name: directory of logs.
        :param threading: whether use multiple threading process.
        :param multiprocessing: whether use multiprocessing module.
        :param console_mode: write the logs on console if True.
        :param console_level: the lowest level for logs on console.
        :param file_mode: write the logs into local file if True.
        :param file_level: the lowest level for logs on file.
        :param covered: re-write the logs to file if True.
        """
        self.filename = filename
        self.dir_name = dir_name
        self.threading = threading
        self.multiprocessing = multiprocessing
        self.console_mode = console_mode
        self.console_level = console_level
        self.file_mode = file_mode
        self.file_level = file_level
        self.write_mode = 'w' if covered else 'a'

    def set_logger(self):
        self.logger = logging.getLogger(self.filename)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = self.generate_formatter()
        if self.console_mode:
            self.generate_console_handler()
        if self.file_mode:
            self.generate_file_handler()
            self.generate_log_file()
        return self.logger

    @contextmanager
    def timer(self, message):
        start = time.time()
        yield self.logger.info(f"{message} - done in {time.time() - start}")

    def generate_console_handler(self):
        handler = logging.StreamHandler()
        handler.setLevel(self.console_level)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def generate_file_handler(self):
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        handler = logging.FileHandler(
            filename=f"{self.dir_name}/{self.filename}.txt", encoding="utf-8"
        )
        handler.setLevel(self.file_level)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def generate_log_file(self):
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        initials = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(f"{self.dir_name}/{self.filename}.txt", self.write_mode) as f:
            f.write(f"\n{'=' * 72}\n{initials}\n{'=' * 72}\n")

    def generate_formatter(self):
        if self.threading and self.multiprocessing:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(process)d - %(thread)d - %(levelname)s:\n %(message)s"
            )
        elif self.threading:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(thread)d - %(levelname)s:\n %(message)s"
            )
        elif self.multiprocessing:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(process)d - %(levelname)s:\n %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s:\n %(message)s"
            )
        return formatter