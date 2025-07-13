import re
import string
from collections import defaultdict, Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization
    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id


def trim_input_ids(input_ids: torch.tensor, pad_token_id, mask_token_id, num_masks: int):
    """
    Trim a sequence of input ids by removing all padding tokens and keeping at most a specific number of mask tokens.
    :param input_ids: the sequence of input token ids
    :param pad_token_id: the id of the pad token
    :param mask_token_id: the id of the mask tokens
    :param num_masks: the number of masks to keeps
    :return: the trimmed sequence of input ids
    """
    assert input_ids.shape[0] == 1
    input_ids_without_pad = [x for x in input_ids[0] if x != pad_token_id]

    trimmed_input_ids = []
    mask_count = 0
    for input_id in input_ids_without_pad:
        if input_id == mask_token_id:
            if mask_count >= num_masks:
                continue
            mask_count += 1
        trimmed_input_ids.append(input_id)

    return torch.tensor([trimmed_input_ids], dtype=torch.long, device=input_ids.device)
