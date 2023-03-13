# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"


class Item(object):
    """
    Class representing an item in a transcript.

    """

    def __init__(self, start_time, end_time, content):
        """
        Constructor.

        Args:
            start_time: The start time of the item in seconds (string) e.g. "75.24"
            end_time: The end time of the item in seconds (string)
            content: The content of the item (string)

        """

        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.content = content
