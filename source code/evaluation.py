import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

nltk.download('wordnet')
nltk.download('omw-1.4')

def nltk_sentence_bleu(hypotheses, references, order=4):
    refs = []
    count = 0
    total_score = 0.0
    total_score1 = 0.0
    total_score2 = 0.0
    total_score3 = 0.0
    total_score4 = 0.0

    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        if len(hyp) < order:
            continue
        else:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
            score1 = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=cc.method4)
            score2 = sentence_bleu([ref], hyp, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
            score3 = sentence_bleu([ref], hyp, weights=(0, 0, 1, 0), smoothing_function=cc.method4)
            score4 = sentence_bleu([ref], hyp, weights=(0, 0, 0, 1), smoothing_function=cc.method4)

            total_score += score
            total_score1 += score1
            total_score2 += score2
            total_score3 += score3
            total_score4 += score4

            count += 1
    avg_score = total_score / count
    avg_score1 = total_score1 / count
    avg_score2 = total_score2 / count
    avg_score3 = total_score3 / count
    avg_score4 = total_score4 / count

    # print('avg_score: %.4f' % avg_score)
    print('1-BLEU: %.4f' % avg_score1)
    print('2-BLEU: %.4f' % avg_score2)
    print('3-BLEU: %.4f' % avg_score3)
    print('4-BLEU: %.4f' % avg_score4)
    return avg_score

# def nltk_sentence_bleu(hypotheses, references, order=4):
#     refs = []
#     count = 0
#     total_score = 0.0
#     # cc = SmoothingFunction()
#     for hyp, ref in zip(hypotheses, references):
#         hyp = hyp.split()
#         ref = ref.split()
#         refs.append([ref])
#         if len(hyp) < order:
#             continue
#         else:
#             # score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
#             score = meteor_score([ref], hyp)
#             total_score += score
#             count += 1
#     avg_score = total_score / count
#     print('avg_score: %.4f' % avg_score)
#     return avg_score