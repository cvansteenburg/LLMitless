import re
from types import MappingProxyType
from typing import Callable, cast

from bs4 import BeautifulSoup as soup
from bs4.element import Tag
from markdownify import markdownify as md


def postmark_html_strip_address(html_content: str) -> str:
    souped_html = soup(html_content, "html.parser")

    for header_div in souped_html.find_all("div", class_="gmail_attr"):
        cast(Tag, header_div).decompose()

    return str(souped_html)


# Markdownify doesn't bring lists to a newline after bolded text
# eg. **foo***list should be **foo**\n*list
def md_lists_to_newline(md_text):
    # Pattern to find headers or bold text in markdown followed immediately by a list item
    pattern = r"(\n(?:\#{1,6} .+|\*\*.+\*\*|\_\_.+\_\_))([ \t]*[\*\-\+] .+)"
    # Replace with the matched text and a newline before the list item
    processed_md = re.sub(pattern, r"\1\n\2", md_text, flags=re.MULTILINE)

    # For ordered lists following headers or bold text
    ordered_list_pattern = r"(\n(?:\#{1,6} .+|\*\*.+\*\*|\_\_.+\_\_))([ \t]*\d+\. .+)"
    processed_md = re.sub(
        ordered_list_pattern, r"\1\n\2", processed_md, flags=re.MULTILINE
    )

    return processed_md


def md_strip_extra_newlines(md_text):
    return re.sub(r"\n{3,}", "\n\n", md_text)


def md_strip_extra_nbsp(md_text):
    return re.sub(r"[\xa0\u200c]", "", md_text)


def bs_html_to_text(html_body: str) -> str:
    return soup(html_body, "html5lib").get_text()


def markdownify_html_to_md(html_body: str) -> str:
    pre_processed_html = postmark_html_strip_address(html_body)

    md_text = md(pre_processed_html, strip=["a", "img", "tr", "td"]).strip()

    md_text = md_lists_to_newline(md_text)

    md_text = md_strip_extra_newlines(md_text)

    md_text = md_strip_extra_nbsp(md_text)

    return md_text


def string_input_passthrough(input: str) -> str:
    return input


# NOTE: Fns must take only one arg for use in LLM chains. If more are needed, use partial
_parse_fns: dict[str, Callable[[str], str]] = {
    "bs_html_to_text": bs_html_to_text,
    "markdownify_html_to_md": markdownify_html_to_md,
    "passthrough": string_input_passthrough,
}

PARSE_FNS = MappingProxyType(_parse_fns)
