import unittest

from lls.intersection_readout import language_mentions


class LanguageMentionsTest(unittest.TestCase):
    def test_detects_each_language_and_both_without_substring_matches(self):
        languages = ["Spanish", "Russian"]
        self.assertEqual(language_mentions("Speak Spanish or Russian.", languages),
                         {"Spanish": True, "Russian": True})
        self.assertEqual(language_mentions("español / русский язык", languages),
                         {"Spanish": True, "Russian": True})
        self.assertTrue(all(language_mentions("西班牙语和俄语", languages).values()))
        self.assertTrue(all(language_mentions("Español y ruso", languages).values()))
        self.assertFalse(any(language_mentions("Prussian blue", languages).values()))


if __name__ == "__main__":
    unittest.main()
