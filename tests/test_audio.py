import unittest
from milo.audio import VoiceIO


class AudioTests(unittest.TestCase):
    def test_noise_annotations_are_ignored(self):
        for text in ('(wind blowing)', '[music]', '(background noise)', '[inaudible]', '*gunshot*'):
            self.assertEqual(VoiceIO._clean_transcription(text), '')

    def test_real_speech_is_kept(self):
        self.assertEqual(VoiceIO._clean_transcription('Hello Milo, can you hear me?'), 'Hello Milo, can you hear me?')


if __name__ == '__main__': unittest.main()
