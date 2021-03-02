# -*- coding: utf-8 -*-

# hellolzc@2018/12/19
# This code is extract from Language Acquisition Research Tools in Python http://pylangacq.org/

"""Interfacing with CHAT data files."""

import sys
import os
import fnmatch
import re
import tempfile
import uuid
from pprint import pformat
from collections import Counter
from itertools import chain
from functools import wraps

# from pylangacq.measures import get_MLUm, get_MLUw, get_TTR, get_IPSyn
from pylangacq_modified.util import (ENCODING, CLITIC,
                            get_participant_code, convert_date_to_tuple,
                            clean_utterance, clean_word, get_lemma_from_mor,
                            get_time_marker)


if sys.version_info[0] == 2:  # pragma: no coverage
    from io import open
    unicode_ = unicode  # noqa F821 (undefined name 'unicode' in python >= 3)
else:  # pragma: no coverage
    unicode_ = str


# NOTE remove this if statement when dropping python 2.7 support
if sys.version_info[:2] >= (3, 4):
    # 'U' deprecated since python 3.4, to removed in python 4.0
    # https://docs.python.org/3/library/functions.html#open
    _OPEN_MODE = 'r'
else:
    _OPEN_MODE = 'rU'



# delete class Reader for reading multiple CHAT files. hellolzc@2018/12/18

class SingleReader(object):
    """A class for reading a single CHAT file."""

    def __init__(self, filename, encoding=ENCODING):

        self.encoding = encoding

        if not isinstance(filename, (str, unicode_)):
            raise ValueError('filename must be str')

        self._filename = os.path.abspath(filename)

        if not os.path.isfile(self._filename):
            raise FileNotFoundError(self._filename)  # noqa F821 (py2 compat)

        self._headers = self._get_headers()
        self._index_to_tiers = self._get_index_to_tiers()
        self.tier_markers = self._tier_markers()

        self._part_of_speech_tags = None

        # list of (partcipant, list of tagged sents)
        self._all_tagged_sents = self._create_all_tagged_sents()

        # for MLUw() and TTR()
        self.words_to_ignore = {'', '!', '+...', '.', ',', '?', '‡',
                                '„', '0', CLITIC}

        # for MLUm()
        self.pos_to_ignore = {'', '!', '+...', '0', '?', 'BEG'}

    def __len__(self):
        return len(self._index_to_tiers)

    def number_of_utterances(self, participant=None, exclude=None):
        return len(self.utterances(participant=participant, exclude=exclude))

    def filename(self):
        return self._filename

    def cha_lines(self):
        """A generator of lines in the CHAT file,
        with the tab-character line continuations undone.
        """
        previous_line = ''

        for line in open(self._filename, mode=_OPEN_MODE,
                         encoding=self.encoding):
            previous_line = previous_line.strip()
            current_line = line.rstrip()  # don't remove leading \t

            if not current_line:
                continue

            if current_line.startswith('%xpho:') or \
                    current_line.startswith('%xmod:'):
                current_line = current_line.replace('%x', '%', 1)

            if previous_line and current_line.startswith('\t'):
                previous_line = u'{} {}'.format(
                    previous_line, current_line.strip())  # strip \t
            elif previous_line:
                yield previous_line
                previous_line = current_line
            else:  # when it's the very first line
                previous_line = current_line
        yield previous_line  # don't forget the very last line!

    def _tier_markers(self):
        """Determine what the %-tiers are."""
        result = set()
        for tiermarkers_to_tiers in self._index_to_tiers.values():
            for tier_marker in tiermarkers_to_tiers.keys():
                if tier_marker.startswith('%'):
                    result.add(tier_marker)
        return result

    def index_to_tiers(self):
        """
        Return a dict of utterances and the corresponding tiers.

        :return: A dict where key is utterance index (starting from 0)
            and value is a dict,
            where key is tier marker and value is the line as str. For example,
            two key-value pairs in the output dict may look like this::

                 1537: {'%gra': '1|2|MOD 2|0|INCROOT 3|2|PUNCT',
                        '%mor': 'n|tapioca n|finger .',
                        'CHI': 'tapioca finger . [+ IMIT]'},
                 1538: {'%gra': '1|0|INCROOT 2|1|PUNCT',
                        '%mor': 'n|cracker .',
                        'MOT': 'cracker .'}

        :rtype: dict(int: dict(str: str))
        """
        return self._index_to_tiers

    def _get_index_to_tiers(self):
        result_with_collapses = {}
        index_ = -1  # utterance index (1st utterance is index 0)
        utterance = None

        for line in self.cha_lines():
            if line.startswith('@'):
                continue

            line_split = line.split()

            if line.startswith('*'):
                index_ += 1
                participant_code = line_split[0].lstrip('*').rstrip(':')
                utterance = ' '.join(line_split[1:])
                result_with_collapses[index_] = {participant_code: utterance}

            elif utterance and line.startswith('%'):
                tier_marker = line_split[0].rstrip(':')
                result_with_collapses[index_][tier_marker] = \
                    ' '.join(line_split[1:])

        return result_with_collapses
        # The meaning of [x 4] is repeating of specific word, not sentence hellolzc@20181219
        # # handle collapses such as [x 4]
        # result_without_collapses = {}
        # new_index = -1  # utterance index (1st utterance is index 0)
        # collapse_pattern = re.compile(r'\[x \d+?\]')  # e.g., "[x <number(s)>]"
        # number_regex = re.compile(r'\d+')

        # for old_index in range(len(result_with_collapses)):
        #     tier_dict = result_with_collapses[old_index]
        #     participant_code = get_participant_code(tier_dict.keys())
        #     utterance = tier_dict[participant_code]

        #     try:
        #         collapse_str = collapse_pattern.search(utterance).group()
        #         collapse_number = int(number_regex.findall(collapse_str)[0])
        #     except (AttributeError, ValueError):
        #         collapse_number = 1

        #     for i in range(collapse_number):
        #         new_index += 1
        #         result_without_collapses[new_index] = tier_dict

        # return result_without_collapses

    def headers(self):
        """
        Return the headers as a dict.

        :return: A dict of headers of the CHAT file.

        The keys are the header names
            as str (e.g., 'Begin', 'Participants', 'Date'). The header entry is
            the content for the respective header name.

        For the head 'Participants', the entry is a dict where the keys are the
            participant codes (e.g., 'CHI', 'MOT') and the value is a dict of
            information for the respective participant code. The keys of the
            information are as follows:

                * participant_label
                * language
                * corpus
                * code
                * age
                * sex
                * group
                * SES
                * role
                * education
                * custom

        :rtype: dict(str: dict)
        """
        return self._headers

    def _get_headers(self):
        headname_to_entry = {}

        for line in self.cha_lines():

            if line.startswith('@Begin') or line.startswith('@End'):
                continue

            if not line.startswith('@'):
                continue

            # find head, e.g., "Languages", "Participants", "ID" etc
            head, _, line = line.partition('\t')
            line = line.strip()
            head = head.lstrip('@')  # remove beginning "@"
            head = head.rstrip(':')  # remove ending ":", if any

            if head == 'Participants':
                headname_to_entry['Participants'] = {}

                participants = line.split(',')

                for participant in participants:
                    participant = participant.strip()
                    code, _, participant_label = participant.partition(' ')
                    participant_name, _, participant_role = \
                        participant_label.partition(' ')
                    # code = participant code, e.g. CHI, MOT
                    headname_to_entry['Participants'][code] = \
                        {'participant_name': participant_name}

            elif head == 'ID':
                participant_info = line.split('|')[: -1]
                # final empty str removed

                code = participant_info[2]
                # participant_info contains these in order:
                #   language, corpus, code, age, sex, group, SES, role,
                #   education, custom

                del participant_info[2]  # remove code info (3rd in list)
                participant_info_heads = ['language', 'corpus', 'age', 'sex',
                                          'group', 'SES', 'participant_role',
                                          'education', 'custom']
                head_to_info = dict(zip(participant_info_heads,
                                        participant_info))

                headname_to_entry['Participants'][code].update(head_to_info)

            elif head == 'Date':
                if 'Date' not in headname_to_entry:
                    headname_to_entry['Date'] = []
                headname_to_entry['Date'].append(line)

            else:
                headname_to_entry[head] = line

        return headname_to_entry

    def participants(self):
        """
        Return the participant information as a dict.

        :return: A dict of participant information based on the @ID lines,
            where the key is the participant code, and the value is a dict of
            info for the participant. Example::

                {'CHI': {'SES': '',
                         'age': '1;6.',
                         'corpus': 'Brown',
                         'custom': '',
                         'education': '',
                         'group': '',
                         'language': 'eng',
                         'participant_label': 'Eve Target_Child',
                         'role': 'Target_Child',
                         'sex': 'female'},
                 'COL': {'SES': '',
                         'age': '',
                         'corpus': 'Brown',
                         'custom': '',
                         'education': '',
                         'group': '',
                         'language': 'eng',
                         'participant_label': 'Colin Investigator',
                         'role': 'Investigator',
                         'sex': ''},
                 'MOT': {'SES': '',
                         'age': '',
                         'corpus': 'Brown',
                         'custom': '',
                         'education': '',
                         'group': '',
                         'language': 'eng',
                         'participant_label': 'Sue Mother',
                         'role': 'Mother',
                         'sex': ''},
                 'RIC': {'SES': '',
                         'age': '',
                         'corpus': 'Brown',
                         'custom': '',
                         'education': '',
                         'group': '',
                         'language': 'eng',
                         'participant_label': 'Richard Investigator',
                         'role': 'Investigator',
                         'sex': ''}}
        """
        try:
            return self._headers['Participants']
        except KeyError:
            return {}

    def participant_codes(self):
        """
        Return the set of participant codes (e.g., `{'CHI', 'MOT', 'FAT'}`).
        """
        try:
            return set(self._headers['Participants'].keys())
        except KeyError:
            return set()

    def languages(self):
        """
        Return the list of the languages involved based on the @Languages
        header.
        """
        languages_list = []

        try:
            languages_line = self._headers['Languages']
        except KeyError:
            pass
        else:
            for language in languages_line.split(','):
                language = language.strip()
                if language:
                    languages_list.append(language)

        return languages_list

    def dates_of_recording(self):
        """
        Return the date of recording as a tuple of (*year*, *month*, *day*).
        If any errors arise (e.g., there's no date), return ``None``.

        :rtype: list(tuple(int, int, int))
        """
        try:
            dates = self._headers['Date']
        except KeyError:
            return None

        return [convert_date_to_tuple(date) for date in dates]

    def date_of_birth(self):
        """
        Return the dates of birth as
        dict(participant code: (*year*, *month*, *day*)).
        If no date of birth is given for a participant,
        the value is ``None`` instead of the tuple.

        :rtype: dict(str: (int, int, int))
        """
        header_keys = self._headers.keys()
        participant_to_date = {}

        for header in header_keys:
            if not header.startswith('Birth of'):
                continue

            # e.g., header is 'Birth of CHI', participant is 'CHI'
            _, _, participant = header.split()
            date_str = self._headers[header]

            participant_to_date[participant] = convert_date_to_tuple(date_str)

        for participant in self.participants():
            if participant not in participant_to_date:
                participant_to_date[participant] = None

        return participant_to_date

    def age(self, participant='CHI', months=False):
        """
        Return the age of *participant* as a tuple or a float.

        :param participant: The participant specified, default to ``'CHI'``

        :param months: If True (default: False), return age in months.

        :return: The age as a 3-tuple of (years, months, days).
            If any errors arise (e.g., there's no age), ``None`` is returned.
            If *month* is True (default: False),
            return a float as age in months instead.

        :rtype: tuple or float
        """
        try:
            age_ = self._headers['Participants'][participant]['age']

            year_str, _, month_day = age_.partition(';')
            month_str, _, day_str = month_day.partition('.')

            year_int = int(year_str) if year_str.isdigit() else 0
            month_int = int(month_str) if month_str.isdigit() else 0
            day_int = int(day_str) if day_str.isdigit() else 0

            if months:
                return year_int * 12 + month_int + day_int / 30
            else:
                return year_int, month_int, day_int
        except (KeyError, IndexError, ValueError):
            return None

    def utterances(self, participant=None, exclude=None, clean=True, time_marker=False):
        """
        Return a list of the utterances by *participant*
        as (*participant*, *utterance*) pairs.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.

        :param clean: Whether to filter away the CHAT annotations in the
            utterance; default to ``True``.
        
        :param time_marker: Whether to include the timer marker in the
            utterance; default to ``False``.
        """
        output = []
        participants = self._determine_participants(participant, exclude)

        for i in range(len(self)):
            tiermarker_to_line = self._index_to_tiers[i]

            for tier_marker in tiermarker_to_line.keys():
                if tier_marker in participants:
                    line = tiermarker_to_line[tier_marker]
                    if clean:
                        if time_marker:
                            try:
                                time_marker = get_time_marker(line)
                            except ValueError as e:
                                print("At line %d in file %s"%(i, self.filename()))
                                print(e)
                                raise Exception
                            output.append((tier_marker, clean_utterance(line), time_marker))
                        else:
                            output.append((tier_marker, clean_utterance(line)))
                    else:
                        output.append((tier_marker, line))
                    break
        return output

    def _determine_participants(self, participant, exclude):
        """Determine the target participants.

        Parameters
        ----------
        participant : str or iterable of str
            Participants to include.
            If unspecified or ``None``, all participant codes are included.
        exclude : str or iterable of str
            Participants to exclude.
            If unspecified or ``None``, no participant codes are excluded.

        Returns
        -------
        set of str
        """
        if participant is None and exclude is None:
            return self.participant_codes()

        if participant is None:
            include_participants = self.participant_codes()
        elif isinstance(participant, (str, unicode_)):
            include_participants = {participant}
        elif hasattr(participant, '__iter__'):
            include_participants = set(participant)
        else:
            raise TypeError('"participant" should be either str or '
                            'an iterable of str: {}'
                            .format(repr(participant)))

        if exclude is None:
            exclude_participants = set()
        elif isinstance(exclude, (str, unicode_)):
            exclude_participants = {exclude}
        elif hasattr(exclude, '__iter__'):
            exclude_participants = set(exclude)
        else:
            raise TypeError('"exclude" should be either str or '
                            'an iterable of str: {}'
                            .format(repr(exclude)))

        return {p for p in self.participant_codes()
                if p in include_participants and p not in exclude_participants}

    def words(self, participant=None, exclude=None):
        """
        Return a list of words by *participant*.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.
        """
        return self._get_words(participant=participant, exclude=exclude,
                               tagged=False, sents=False)

    def tagged_words(self, participant=None, exclude=None):
        """
        Return a list of tagged words by *participant*.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.
        """
        return self._get_words(participant=participant, exclude=exclude,
                               tagged=True, sents=False)

    def sents(self, participant=None, exclude=None):
        """
        Return a list of sents by *participant*.

        (utterances = sents in NLTK terminology)

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.
        """
        return self._get_words(participant=participant, exclude=exclude,
                               tagged=False, sents=True)

    def tagged_sents(self, participant=None, exclude=None):
        """
        Return a list of tagged sents by *participant*.

        (utterances = sents in NLTK terminology)

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.
        """
        return self._get_words(participant=participant, exclude=exclude,
                               tagged=True, sents=True)

    def _get_words(self, participant=None, exclude=None, tagged=True,
                   sents=True):
        """
        Extract words for the specified participant(s).

        The representation of "word" depends on whether ``tagged`` is True, and
        is based to some extent on the NLTK conventions.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.

        :param tagged:

        If ``tagged`` is True, a word is a 4-tuple of
            (*word*, *PoS*, *mor*, *gra*), where:

            * *word* is str
            * *PoS* is part-of-speech tag as str,
              forced to be in uppercase following NLTK
            * *mor* is morphological information as str
            * *gra* is grammatical relation, as a 3-tuple of
              (*self-position*, *head-position*, *relation*),
              with the data type (int, int, str).

        An example word with this representation::

        ('thought', 'V', 'think&PAST', (3, 0, 'ROOT'))

            where in the original data, "thought" is the transcription,
            %mor has "v|think&PAST", and %gra is "3|0|ROOT"

        This word representation is an extension of NLTK, where a tagged word
            is typically a 2-tuple of (word, PoS).

        If PoS, mor, gra correspond to a "word" that is a clitic (due to the
            tilde in the original CHAT data), then word is 'CLITIC'.

        If ``tagged`` is False, a word is simply the word (as a str) from the
            transcription. If the word is 'CLITIC", it is not included in the
            returned generator.

        :param sents: If ``sents`` (using NLTK terminology) is True,
            words from the same utterance (= "sentence") are grouped
            together into a list which is in turn yielded. Otherwise,
            individual words are directly yielded without utterance structure.
        """
        result_list = []
        participants = self._determine_participants(participant, exclude)

        if sents:
            add_function = lambda result_, sent_: result_.append(sent_)
        else:
            add_function = lambda result_, sent_: result_.extend(sent_)

        if tagged:
            sent_to_add = lambda sent_: sent_
        else:
            sent_to_add = lambda sent_: [x[0] for x in sent_ if x[0] != CLITIC]

        for participant_code, tagged_sent in self._all_tagged_sents:
            if participant_code not in participants:
                continue

            add_function(result_list, sent_to_add(tagged_sent))

        return result_list

    def _create_all_tagged_sents(self):
        result_list = []

        for i in range(self.number_of_utterances()):
            tiermarker_to_line = self._index_to_tiers[i]
            participant_code = get_participant_code(tiermarker_to_line.keys())

            # get the plain words from utterance tier
            utterance = clean_utterance(
                tiermarker_to_line[participant_code])
            words = utterance.split()

            # %mor tier
            clitic_indices = []  # indices at the word items
            clitic_count = 0

            mor_items = []
            if '%mor' in tiermarker_to_line:
                mor_split = tiermarker_to_line['%mor'].split()

                for j, item in enumerate(mor_split):
                    tilde_count = item.count('~')

                    if tilde_count:
                        item_split = item.split('~')

                        for k in range(tilde_count):
                            clitic_indices.append(clitic_count + j + k + 1)
                            clitic_count += 1

                            mor_items.append(item_split[k])

                        mor_items.append(item_split[-1])
                    else:
                        mor_items.append(item)

            if mor_items and ((len(words) + clitic_count) != len(mor_items)):
                message = 'cannot align the utterance and %mor tiers:\n' + \
                          'Filename: {}\nTiers --\n{}\n' + \
                          'Cleaned-up utterance --\n{}'
                raise ValueError(message.format(
                    self.filename(),
                    pformat(tiermarker_to_line), utterance))

            # %gra tier
            gra_items = []
            if '%gra' in tiermarker_to_line:
                for item in tiermarker_to_line['%gra'].split():
                    # an item is a string like '1|2|SUBJ'

                    item_list = []
                    for element in item.split('|'):
                        try:
                            converted_element = int(element)
                        except ValueError:
                            converted_element = element

                        item_list.append(converted_element)

                    gra_items.append(tuple(item_list))

            if mor_items and gra_items and \
                    (len(mor_items) != len(gra_items)):
                raise ValueError('cannot align the %mor and %gra tiers:\n{}'
                                 .format(pformat(tiermarker_to_line)))

            # utterance tier
            if mor_items and clitic_count:
                word_iterator = iter(words)
                utterance_items = [''] * len(mor_items)

                for j in range(len(mor_items)):
                    if j in clitic_indices:
                        utterance_items[j] = CLITIC
                    else:
                        utterance_items[j] = next(word_iterator)
            else:
                utterance_items = words

            # determine what to yield (and how) to create the generator
            if not mor_items:
                mor_items = [''] * len(utterance_items)
            if not gra_items:
                gra_items = [''] * len(utterance_items)

            sent = []

            for word, mor, gra in zip(utterance_items, mor_items, gra_items):
                pos, _, mor = mor.partition('|')

                output_word = (clean_word(word), pos.upper(), mor, gra)
                # pos in uppercase follows NLTK convention
                sent.append(output_word)

            result_list.append((participant_code, sent))

        return result_list

    def word_frequency(self, participant=None, exclude=None, keep_case=True):
        """
        Return the word frequency Counter dict for *participant*.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.

        :param keep_case: If *keep_case* is True (the default), case
            distinctions are kept and word tokens like "the" and "The" are
            treated as distinct types. If *keep_case* is False, all case
            distinctions are collapsed, with all word tokens forced to be in
            lowercase.
        """
        output = Counter()

        if keep_case:
            for word in self.words(participant=participant, exclude=exclude):
                output[word] += 1
        else:
            for word in self.words(participant=participant, exclude=exclude):
                output[word.lower()] += 1

        return output

    def part_of_speech_tags(self, participant=None, exclude=None):
        """
        Return the set of part-of-speech tags in the data for *participant*.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.
        """
        output_set = set()
        tagged_words = self.tagged_words(participant=participant,
                                         exclude=exclude)

        for tagged_word in tagged_words:
            pos = tagged_word[1]
            output_set.add(pos)

        return output_set

    def word_ngrams(self, n, participant=None, exclude=None, keep_case=True):
        """
        Return a Counter dict of *n*-grams (as an *n*-tuple of words)
        for *participant*.

        :param participant: The participant(s) of interest (default is all
            participants if unspecified). This parameter is flexible.
            Set it to be ``'CHI'`` for the target child only, for example.
            If multiple participants are desired, this parameter can take
            a sequence such as ``{'CHI', 'MOT'}`` to pick the participants in
            question. Underlyingly, this parameter actually performs
            regular expression matching
            (so passing ``'CHI'`` to this parameter is an
            exact match for the participant code ``'CHI'``, for instance).
            For child-directed speech (i.e., targeting all participant
            except ``'CHI'``), use ``^(?!.*CHI).*$``.

        :param keep_case: If *keep_case* is True (the default), case
            distinctions are kept and word tokens like "the" and "The" are
            treated as distinct types. If *keep_case* is False, all case
            distinctions are collapsed, with all word tokens forced to be in
            lowercase.
        """
        if (type(n) is not int) or (n < 1):
            raise ValueError('n must be a positive integer: %r' % n)

        if n == 1:
            return self.word_frequency(participant=participant,
                                       exclude=exclude,
                                       keep_case=keep_case)

        sents = self.sents(participant=participant, exclude=exclude)
        output_counter = Counter()

        for sent in sents:
            if len(sent) < n:
                continue
            if not keep_case:
                sent = [word.lower() for word in sent]
            ngram_list = zip(*[sent[i:] for i in range(n)])
            output_counter.update(ngram_list)

        return output_counter

# delete some codes about pylangacq.measures

    def search(self, search_item, participant=None, exclude=None,
               match_entire_word=True, lemma=False,
               output_tagged=True, output_sents=True):
        return self._search(search_item, participant=participant,
                            exclude=exclude,
                            match_entire_word=match_entire_word, lemma=lemma,
                            concordance=False, output_tagged=output_tagged,
                            output_sents=output_sents)

    def concordance(self, search_item, participant=None, exclude=None,
                    match_entire_word=True, lemma=False):
        return self._search(search_item, participant=participant,
                            exclude=exclude,
                            match_entire_word=match_entire_word, lemma=lemma,
                            concordance=True)

    def _search(self, search_item, participant=None, exclude=None,
                match_entire_word=True, lemma=False, concordance=False,
                output_tagged=True, output_sents=True):
        taggedsent_charnumber_list = []
        # = list of (tagged_sent, char_number)

        # set up the match function
        if match_entire_word or lemma:
            match_function = lambda search_, test_: search_ == test_
        else:
            match_function = lambda search_, test_: search_ in test_

        tagged_sents = self.tagged_sents(participant=participant,
                                         exclude=exclude)

        for tagged_sent in tagged_sents:
            for i, tagged_word in enumerate(tagged_sent):
                word, pos, mor, rel = tagged_word

                # test_item targets word by default
                # if the "lemma" parameter is True,
                # then shift test_item to lemma extract from mor
                test_item = word
                if lemma:
                    test_item = get_lemma_from_mor(mor)

                # run the match test
                # if match, keep the tagged_sent and compute char_number
                # char_number = the number of characters that would precede the
                #               target **word** if sent was represented as str
                #               (as is the case when "concordance" is True)
                if match_function(search_item, test_item):

                    preceding_words = [tagged_sent[k][0] for k in range(i)]
                    preceding_words = [w for w in preceding_words
                                       if w != CLITIC]  # remove CLITIC
                    char_number = (sum([len(w) for w in preceding_words]) +
                                   len(preceding_words) - 1)  # plus spaces
                    taggedsent_charnumber_list.append((tagged_sent,
                                                       char_number))

        if not taggedsent_charnumber_list:  # if empty
            return taggedsent_charnumber_list

        if not concordance:

            if output_sents:
                add_function = lambda result_, sent_: result_.append(sent_)
            else:
                add_function = lambda result_, sent_: result_.extend(sent_)

            if output_tagged:
                sent_to_add = lambda sent_: sent_
            else:
                sent_to_add = lambda sent_: [x[0]
                                             for x in sent_ if x[0] != CLITIC]

            result_list = []

            for tagged_sent, _ in taggedsent_charnumber_list:
                add_function(result_list, sent_to_add(tagged_sent))

            return result_list
        else:
            max_char_number = max([n for _, n in taggedsent_charnumber_list])
            result_list = []

            for tagged_sent, char_number in taggedsent_charnumber_list:
                sent = [word_ for word_, _, _, _ in tagged_sent
                        if word_ != CLITIC]
                sent_str = (' ' * (max_char_number - char_number) +
                            ' '.join(sent))
                result_list.append(sent_str)

            return result_list
