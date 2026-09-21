import unittest
from milo.llm import LocalModel


class LLMTests(unittest.TestCase):
    def test_content(self):
        response={'choices':[{'message':{'content':'Hello'}}]}
        self.assertEqual(LocalModel._message_text(response['choices'][0]['message']), 'Hello')

    def test_reasoning_content_fallback(self):
        message={'content':'', 'reasoning_content':'Fallback answer'}
        self.assertEqual(LocalModel._message_text(message), 'Fallback answer')

    def test_list_content(self):
        message={'content':[{'type':'text','text':'Hello '},{'type':'text','text':'world'}]}
        self.assertEqual(LocalModel._message_text(message), 'Hello world')

    def test_think_block_removed(self):
        message={'content':'<think>hidden</think>Visible'}
        self.assertEqual(LocalModel._message_text(message), 'Visible')


if __name__ == '__main__': unittest.main()
