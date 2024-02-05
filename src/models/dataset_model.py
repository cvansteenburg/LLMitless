from enum import StrEnum


class DatasetFileFormatNames(StrEnum):
    HTML = "html-message.html"
    JSON = "json-body.json"
    RAW = "raw-source.txt"
    TXT = "text-message.txt"