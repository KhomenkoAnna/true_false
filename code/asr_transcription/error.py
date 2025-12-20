import collections
import jiwer
import pandas as pd

from jiwer.alignment import visualize_alignment
from typing import Union, Optional
from jiwer.process import WordOutput, CharacterOutput
from collections import defaultdict

def collect_error_counts(output: Union[WordOutput, CharacterOutput]):
    """
    Retrieve three dictionaries, which count the frequency of how often
    each word or character was substituted, inserted, or deleted.
    The substitution dictionary has, as keys, a 2-tuple (from, to).
    The other two dictionaries have the inserted/deleted words or characters as keys.

    Args:
        output: The processed output of reference and hypothesis pair(s).

    Returns:
        (Tuple[dict, dict, dict]): A three-tuple of dictionaries, in the order substitutions, insertions, deletions.
    """
    substitutions = defaultdict(lambda: 0)
    insertions = defaultdict(lambda: 0)
    deletions = defaultdict(lambda: 0)

    for idx, sentence_chunks in enumerate(output.alignments):
        ref = output.references[idx]
        hyp = output.hypotheses[idx]
        sep = " " if isinstance(output, WordOutput) else ""

        for chunk in sentence_chunks:
            if chunk.type == "insert":
                inserted = sep.join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
                insertions[inserted] += 1
            if chunk.type == "delete":
                deleted = sep.join(ref[chunk.ref_start_idx : chunk.ref_end_idx])
                deletions[deleted] += 1
            if chunk.type == "substitute":
                replaced = sep.join(ref[chunk.ref_start_idx : chunk.ref_end_idx])
                by = sep.join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
                substitutions[(replaced, by)] += 1

    return substitutions, insertions, deletions


def visualize_error(
    output: Union[WordOutput, CharacterOutput],
    show_substitutions: bool = True,
    show_insertions: bool = True,
    show_deletions: bool = True,
    top_k: Optional[int] = None,
):
    """
    Visualize which words (or characters), and how often, were substituted, inserted, or deleted.

    Args:
        output: The processed output of reference and hypothesis pair(s).
        show_substitutions: If true, visualize substitution errors.
        show_insertions: If true, visualize insertion errors.
        show_deletions: If true, visualize deletion errors.
        top_k: If set, only visualize the k most frequent errors.

    Returns:
         (str): A string which visualizes the words/characters and their frequencies.

    Example:
        The code snippet
        ```python3
        import jiwer

        out = jiwer.process_words(
            ["short one here", "quite a bit of longer sentence"],
            ["shoe order one", "quite bit of an even longest sentence here"],
        )
        print(jiwer.visualize_error_counts(out))
        ```

        will print the following:

        ```txt
        === SUBSTITUTIONS ===
        short   --> order   = 1x
        longer  --> longest = 1x

        === INSERTIONS ===
        shoe    = 1x
        an even = 1x
        here    = 1x

        === DELETIONS ===
        here = 1x
        a    = 1x
        ```
    """
    s, i, d = collect_error_counts(output)

    def build_list(errors: dict):
        if len(errors) == 0:
            return "none"

        keys = [k for k in errors.keys()]
        keys = sorted(keys, reverse=True, key=lambda k: errors[k])

        if top_k is not None:
            keys = keys[:top_k]

        # we get the maximum length of all words to nicely pad output
        ln = max(len(k) if isinstance(k, str) else max(len(e) for e in k) for k in keys)

        # here we construct the string
        build = ""

        for count, (k, v) in enumerate(
            sorted(errors.items(), key=lambda tpl: tpl[1], reverse=True)
        ):
            if top_k is not None and count >= top_k:
                break

            if isinstance(k, tuple):
                build += f"{k[0]: <{ln}} --> {k[1]:<{ln}} = {v}x\n"
            else:
                build += f"{k:<{ln}} = {v}x\n"

        return build

    output = ""

    if show_substitutions:
        if output != "":
            output += "\n"
        output += "=== SUBSTITUTIONS ===\n"
        output += build_list(s)

    if show_insertions:
        if output != "":
            output += "\n"
        output += "=== INSERTIONS ===\n"
        output += build_list(i)

    if show_deletions:
        if output != "":
            output += "\n"
        output += "=== DELETIONS ===\n"
        output += build_list(d)

    if output[-1:] == "\n":
        output = output[:-1]

    return output


#Error visualisation
data = pd.read_excel(open('/content/Транскрипты.xlsx', 'rb'),
              sheet_name='Транскрипты')

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveSpecificWords(["эм", "аа", "ээ", "мм", "ам", "угу", "-", "=", "..."]),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.ReduceToListOfListOfWords()
])

hyps = data["nemo-fastconformer-ru-rnnt"].tolist() #you can write the name of any used model here
refs = data["clean_transcription"].tolist()

refs_norm = [" ".join(words) for words in transformation(refs)]
hyps_norm = [" ".join(words) for words in transformation(hyps)]

alignment_output = jiwer.process_words(
    refs_norm,
    hyps_norm
)

alignment = visualize_alignment(alignment_output, show_measures=True)
print(alignment)